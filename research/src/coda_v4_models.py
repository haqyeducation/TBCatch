"""V4 model engines. Every learned transformation is fitted within fit_ids only.

No dependencies beyond the V2/V3 local environment. The 261-dimensional
SciPy descriptor follows Sensors' feature families, not LibROSA bit parity.
"""
from pathlib import Path
import copy, json, math, time
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import dct
from scipy.stats import skew, kurtosis
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from . import coda_research as cr
from . import coda_v3 as v3

CLINICAL = v3.CLINICAL
PAPER_CLINICAL = CLINICAL + ['height','weight','temperature']
REFERENCES = {
    'reference_mobile_lr': ('audio','clinical','hr'),
    'reference_mobile_cb': ('audio','clinical','hr'),
    'reference_paper_lr': ('audio','clinical','hr'),
    'reference_paper_cb': ('audio','clinical','hr')}
COMBINATIONS = {**v3.COMBINATIONS, **REFERENCES}
LABELS = {**v3.LABELS, 'reference_mobile_lr':'Sensors-style mobile LR',
    'reference_mobile_cb':'Sensors-style mobile CatBoost',
    'reference_paper_lr':'Sensors-style clinical LR', 'reference_paper_cb':'Sensors-style clinical CatBoost'}


def paper_features(x, sr=16000):
    """13 MFCC + 12 chroma + 4 spectral tracks, nine functionals each.

    FFT2048/win512/hop256; explicit zero-padded Hann frames, HTK mel40,
    soft pitch-class chroma (A440, no tuning), bias-corrected moments.
    These documented DSP choices are not an exact LibROSA reproduction.
    """
    frames=np.lib.stride_tricks.sliding_window_view(np.asarray(x),512)[::256]
    magnitude=np.abs(np.fft.rfft(frames*np.hanning(512),n=2048,axis=1)).T
    power=magnitude**2; freq=np.fft.rfftfreq(2048,1/sr)
    edges=700*(10**(np.linspace(0,2595*np.log10(1+sr/1400),42)/2595)-1)
    bank=np.array([np.maximum(0,np.minimum((freq-edges[j])/max(edges[j+1]-edges[j],1e-8),
        (edges[j+2]-freq)/max(edges[j+2]-edges[j+1],1e-8))) for j in range(40)])
    mfcc=dct(10*np.log10(np.maximum(bank@power,1e-10)),axis=0,norm='ortho')[:13]
    midi=69+12*np.log2(np.maximum(freq,1e-8)/440)
    distance=(midi[None,:]-np.arange(12)[:,None]+6)%12-6
    chromabank=np.maximum(0,1-np.abs(distance));chromabank[:,freq<27.5]=0
    chroma=chromabank@power;chroma/=np.maximum(np.linalg.norm(chroma,axis=0),1e-10)
    mass=np.maximum(magnitude.sum(axis=0),1e-10)
    centroid=(magnitude*freq[:,None]).sum(axis=0)/mass
    bandwidth=np.sqrt((magnitude*(freq[:,None]-centroid)**2).sum(axis=0)/mass)
    rolloff=freq[np.argmax(np.cumsum(magnitude,axis=0)>=.85*mass,axis=0)]
    flatness=np.exp(np.log(np.maximum(power,1e-10)).mean(axis=0))/np.maximum(power.mean(axis=0),1e-10)
    tracks=np.vstack([mfcc,chroma,centroid,bandwidth,rolloff,flatness])
    # Avoid unstable standardized moments on constant trajectories / silence.
    varying=tracks.std(axis=1)>1e-8
    sk=np.zeros(29);ku=np.zeros(29)
    sk[varying]=skew(tracks[varying],axis=1,bias=False)
    ku[varying]=kurtosis(tracks[varying],axis=1,bias=False,fisher=True)
    stats=np.column_stack([tracks.mean(axis=1),tracks.std(axis=1,ddof=1),sk,ku,
        *[np.percentile(tracks,q,axis=1) for q in [10,25,50,75,90]]])
    return np.nan_to_num(stats.ravel(),nan=0.,posinf=0.,neginf=0.).astype(np.float32)


def prepare_features(participants, manifest, cfg):
    import joblib, shutil
    fp=v3.digest({'input':cfg['input_hashes'],'features_code':cr.sha256(__file__),
        'v3_code':cr.sha256(Path(__file__).with_name('coda_v3.py')),
        'base_code':cr.sha256(Path(__file__).with_name('coda_research.py')),
        'settings':{'sr':16000,'duration':.5}})
    folder=Path(cfg['package_root'])/'cache_v4'/fp[:16];folder.mkdir(parents=True,exist_ok=True)
    path=folder/'participant_features.joblib';meta=folder/'complete.json'
    if path.exists() and meta.exists():
        record=json.loads(meta.read_text())
        if record['fingerprint']!=fp or cr.sha256(path)!=record['sha256']:raise RuntimeError('Cache V4 berubah.')
        table=joblib.load(path)
        assert set(table.index)==set(participants.participant)
        print('Memakai cache V4 terverifikasi:',folder,flush=True)
    else:
        basecfg=copy.deepcopy(cfg)
        basecfg['input_hashes']={k:v for k,v in cfg['input_hashes'].items() if k!='baseline_v3_reference'}
        # V3 cache fingerprints must use V3 source, not V4 source.
        basecfg['source_hashes']={k:cr.sha256(Path(cfg['package_root'])/k) for k in ['src/coda_v3.py','src/coda_research.py']}
        table,_=v3.prepare_features(participants,manifest,basecfg)
        bags_paper={pid:[] for pid in table.index};bags_rich={pid:[] for pid in table.index}
        start=time.perf_counter()
        for i,row in enumerate(manifest.sort_values('filename').itertuples(),1):
            x=cr.read_audio(Path(cfg['data_root'])/row.path,cfg)
            bags_paper[row.participant].append(paper_features(x))
            bags_rich[row.participant].append(v3.rich_features(x)[0].astype(np.float32))
            if i%500==0 or i==len(manifest):print(f'V4 fitur batuk {i:,}/{len(manifest):,} | {time.perf_counter()-start:.0f}s',flush=True)
        for name,bags in [('paper',bags_paper),('rich',bags_rich)]:
            values=[np.stack(bags[pid]) for pid in table.index]
            table['_bag_'+name]=pd.Series(values,index=table.index,dtype=object)
            if name=='paper':
                means=pd.DataFrame(np.stack([v.mean(axis=0) for v in values]),index=table.index,
                    columns=[f'paper_mean_{i:03d}' for i in range(261)])
                table=table.join(means)
        v3.atomic_model(path,table)
        record={'fingerprint':fp,'sha256':cr.sha256(path),'n_participants':len(table),
            'n_coughs':len(manifest),'paper_dimension':261,'rich_dimension':423,
            'notice':'Sensors-style SciPy, not exact LibROSA; bags are data containers, never ID/count predictors.'}
        v3.atomic_json(meta,record)
    assert table.index.is_unique and set(table.index)==set(participants.participant)
    for bag,dim in [('_bag_paper',261),('_bag_rich',423)]:
        assert all(v.ndim==2 and v.shape[0]>0 and v.shape[1]==dim and np.isfinite(v).all() for v in table[bag])
    target=Path(cfg['output_dir'])/'participant_features.joblib'
    if not target.exists() or cr.sha256(target)!=cr.sha256(path):shutil.copyfile(path,target)
    v3.atomic_json(Path(cfg['output_dir'])/'feature_provenance.json',record)
    return table,folder


def candidate_grid(cfg):
    result=[]
    if cfg.get('include_v3_candidates',True):
        old=cr.load_config(cfg['package_root']);old.update(json.loads((Path(cfg['package_root'])/'config_v3.json').read_text()))
        for c in v3.candidate_grid(old):
            result.append({**c,'engine':'legacy','v3_candidate':copy.deepcopy(c)})
    for combo,gs in v3.COMBINATIONS.items():
        for rep in cfg['patient_representations'] if 'audio' in gs else ['none']:
            for C in cfg['upgrade_lr_C']:
                for balanced in [False,True]:
                    for pca in cfg['pca_options'] if 'audio' in gs else [None]:
                        result.append(dict(combination=combo,representation=rep,engine='patient',family='lr',fusion='early',
                            C=C,balanced=balanced,pca=pca))
            for j,trial in enumerate(cfg['upgrade_catboost_trials']):
                result.append(dict(combination=combo,representation=rep,engine='patient',family='catboost',fusion='early',
                    params=trial,balanced=j%2==0,pca=None))
        if 'audio' in gs:
            for rep in cfg['bag_representations']:
                for C in cfg['clip_lr_C']:
                    result.append(dict(combination=combo,representation=rep,engine='clip',family='lr',fusion='per_cough',
                        C=C,balanced=False,pca=None))
                for trial in cfg['clip_catboost_trials']:
                    result.append(dict(combination=combo,representation=rep,engine='clip',family='catboost',fusion='per_cough',
                        params=trial,balanced=False,pca=None))
                if len(gs)>1:
                    result.append(dict(combination=combo,representation=rep,engine='stack',family='lr',fusion='crossfit_stack',
                        C=.1,balanced=False,pca=None))
                if combo in cfg['attention_combinations']:
                    result.append(dict(combination=combo,representation=rep,engine='attention',family='attention',fusion='bag_attention',
                        **cfg['attention']))
    # Four predeclared reference searches, selected by Sensors-style inner UAR.
    for combo in REFERENCES:
        family='lr' if combo.endswith('_lr') else 'catboost'
        trials=[dict(C=c,balanced=w) for c in cfg['reference_lr_C'] for w in [False,True]] if family=='lr' else [
            dict(params=t,balanced=j%2==0) for j,t in enumerate(cfg['upgrade_catboost_trials'])]
        for trial in trials:
            result.append(dict(combination=combo,representation='paper',engine='clip',family=family,
                fusion='per_cough',pca=None,**trial))
    for i,c in enumerate(result):c['id']=f'v4_{i:04d}'
    return result


def groups_for(table,candidate):
    if candidate['engine']=='legacy':return v3.groups_for(table,candidate['v3_candidate'])
    groups={}
    for group in COMBINATIONS[candidate['combination']]:
        if group=='clinical':cols=PAPER_CLINICAL if candidate['combination'].startswith('reference_paper') else CLINICAL
        elif group=='hr':cols=['heart_rate']
        elif candidate['engine'] in ['clip','stack','attention']:cols=['_bag_'+candidate['representation']]
        else:
            prefix={'paper_mean':('paper_mean_',),'rich_mean_std':('rich_mean_','rich_std_')}[candidate['representation']]
            cols=[c for c in table if c.startswith(prefix)]
        assert cols and all(c in table for c in cols),(candidate,cols)
        groups[group]=list(cols)
    forbidden={'tb_status','Country','participant','cough_count','sound_prediction_score'}
    assert not set(sum(groups.values(),[]))&forbidden
    return groups


def _audio_preprocessor(x,candidate,seed,weights=None):
    # Imputation/scaling/optional PCA only fitted on caller's training examples.
    steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]
    if candidate.get('pca'):
        steps.append(('pca',PCA(n_components=min(candidate['pca'],len(x)-1,x.shape[1]),
            svd_solver='randomized',random_state=seed)))
    pp=Pipeline(steps)
    args={'scale__sample_weight':weights} if weights is not None else {}
    pp.fit(x,**args)
    return pp


def _classifier(candidate, cfg, seed):
    if candidate['family']=='lr':return LogisticRegression(C=candidate['C'],
        class_weight='balanced' if candidate.get('balanced') else None,max_iter=10000,random_state=seed)
    from catboost import CatBoostClassifier
    return CatBoostClassifier(**candidate['params'],auto_class_weights='Balanced' if candidate.get('balanced') else None,
        random_seed=seed,task_type='CPU',thread_count=cfg['cpu_threads'],verbose=False,allow_writing_files=False)


def _tab_columns(groups):return sum([cols for g,cols in groups.items() if g!='audio'],[])


def fit_model(table,ids,candidate,cfg,seed):
    ids=list(ids);groups=groups_for(table,candidate)
    model=dict(candidate=candidate,groups=groups,fit_ids=ids,engine=candidate['engine'])
    engine=candidate['engine'];t=table.loc[ids];y=t.tb_status.to_numpy(dtype=int)
    if engine=='legacy':
        model['legacy']=v3.fit_model(table,ids,candidate['v3_candidate'],cfg,seed);return model
    if engine=='stack':
        # No in-sample base predictions enter meta training. Entire stack gets
        # cross-fitted again by the enclosing inner / proper-OOF procedure.
        audio=dict(candidate,engine='clip',combination='audio',family='lr',C=.05,balanced=False)
        tabcombo='clinical_hr' if 'clinical' in groups and 'hr' in groups else 'clinical' if 'clinical' in groups else 'hr'
        tab=dict(candidate,engine='patient',combination=tabcombo,representation='none',family='catboost',
            params=cfg['stack_tab_params'],balanced=False)
        oof=np.empty((len(t),2));cv=StratifiedKFold(cfg['stack_folds'],shuffle=True,random_state=seed)
        inner_ids=[]
        for j,(a,b) in enumerate(cv.split(ids,y)):
            train=t.index[a].tolist();val=t.iloc[b]
            am=fit_model(table,train,audio,cfg,seed+j+11);tm=fit_model(table,train,tab,cfg,seed+j+11)
            oof[b]=np.column_stack([predict_raw(am,val),predict_raw(tm,val)])
            inner_ids.append({'train':train,'validation':val.index.tolist()})
        meta=LogisticRegression(C=candidate['C'],max_iter=3000,random_state=seed).fit(_meta(oof),y)
        model.update(audio=fit_model(table,ids,audio,cfg,seed),tab=fit_model(table,ids,tab,cfg,seed),
            meta=meta,stack_crossfit=inner_ids);return model
    cols=_tab_columns(groups);tabpp=cr.make_preprocessor(cols).fit(t[cols]) if cols else None
    tab=tabpp.transform(t[cols]) if cols else np.zeros((len(t),0))
    model.update(tab_columns=cols,tab_preprocessor=tabpp)
    if engine=='patient':
        blocks=[]
        if 'audio' in groups:
            ap=_audio_preprocessor(t[groups['audio']],candidate,seed)
            blocks.append(ap.transform(t[groups['audio']]));model['audio_preprocessor']=ap
        blocks.append(tab);x=np.concatenate(blocks,axis=1)
        model['model']=_classifier(candidate,cfg,seed).fit(x,y);return model
    bagcol=groups['audio'][0];bags=t[bagcol].tolist();counts=np.array([len(b) for b in bags])
    x=np.concatenate(bags);weights=np.repeat(1/counts,counts)
    ap=_audio_preprocessor(x,candidate,seed,weights);model['audio_preprocessor']=ap
    scaled=ap.transform(x)
    if engine=='clip':
        features=np.concatenate([scaled,np.repeat(tab,counts,axis=0)],axis=1)
        model['model']=_classifier(candidate,cfg,seed).fit(features,np.repeat(y,counts),sample_weight=weights)
    elif engine=='attention':
        from .coda_v4_networks import fit_attention
        cuts=np.r_[0,np.cumsum(counts)]
        model['network']=fit_attention([scaled[cuts[j]:cuts[j+1]] for j in range(len(t))],tab,y,candidate,cfg,seed)
    else:raise ValueError(engine)
    return model


def _meta(p):
    p=np.clip(p,1e-5,1-1e-5);z=np.log(p/(1-p))
    return np.column_stack([z,z[:,0]*z[:,1]])


def predict_raw(model,table):
    engine=model['engine'];groups=model['groups']
    if engine=='legacy':return v3.predict_raw(model['legacy'],table)
    if engine=='stack':
        return model['meta'].predict_proba(_meta(np.column_stack([
            predict_raw(model['audio'],table),predict_raw(model['tab'],table)])))[:,1]
    cols=model['tab_columns'];pp=model['tab_preprocessor']
    tab=pp.transform(table[cols]) if cols else np.zeros((len(table),0))
    if engine=='patient':
        blocks=[]
        if 'audio' in groups:blocks.append(model['audio_preprocessor'].transform(table[groups['audio']]))
        blocks.append(tab);return model['model'].predict_proba(np.concatenate(blocks,axis=1))[:,1]
    bags=table[groups['audio'][0]].tolist();counts=np.array([len(b) for b in bags]);cuts=np.r_[0,np.cumsum(counts)]
    x=model['audio_preprocessor'].transform(np.concatenate(bags))
    if engine=='clip':
        p=model['model'].predict_proba(np.concatenate([x,np.repeat(tab,counts,axis=0)],axis=1))[:,1]
        return np.array([p[cuts[j]:cuts[j+1]].mean() for j in range(len(table))])
    from .coda_v4_networks import predict_attention
    return predict_attention(model['network'],[x[cuts[j]:cuts[j+1]] for j in range(len(table))],tab)


def bag_attention_weights(model,table):
    """Diagnostic learned pooling weights, not automatically faithful XAI."""
    if model['engine']!='attention':return None
    from .coda_v4_networks import predict_attention
    bags=[model['audio_preprocessor'].transform(b) for b in table[model['groups']['audio'][0]]]
    cols=model['tab_columns'];tab=model['tab_preprocessor'].transform(table[cols]) if cols else np.zeros((len(table),0))
    return predict_attention(model['network'],bags,tab,weights=True)
