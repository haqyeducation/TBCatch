"""Seven-input nested CV with fold-local preprocessing, calibration and XAI.

Sensors 2026 evaluation structure, independent audio features and fusion search.
No model selection, cutoff choice or learned preprocessing uses outer test.
"""
from __future__ import annotations
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
import joblib
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import dct
from scipy.stats import beta
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from . import coda_research as cr

COMBINATIONS = {'audio':('audio',), 'hr':('hr',), 'clinical':('clinical',),
    'audio_hr':('audio','hr'), 'clinical_hr':('clinical','hr'),
    'audio_clinical':('audio','clinical'), 'audio_clinical_hr':('audio','clinical','hr')}
LABELS = {'audio':'Audio', 'hr':'Denyut', 'clinical':'Kuesioner', 'audio_hr':'Audio + denyut',
    'clinical_hr':'Kuesioner + denyut', 'audio_clinical':'Audio + kuesioner',
    'audio_clinical_hr':'Audio + kuesioner + denyut'}
CLINICAL = ['age','sex','reported_cough_dur','tb_prior','hemoptysis','weight_loss','smoke_lweek','fever','night_sweats']
POLICIES = ['youden','sensitivity90']
_PP_CACHE = {}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.part'); cr.save_json(temp, value); temp.replace(path)


def atomic_model(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.part'); joblib.dump(value, temp, compress=3); temp.replace(path)


def configure(package, run_name, mode='full', project_root=None):
    if mode not in ('full','preflight') or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,70}', run_name):
        raise ValueError('MODE atau RUN_NAME tidak valid.')
    package = Path(package).resolve()
    cfg = cr.load_config(package)
    cfg.update(json.loads((package/'config_v3.json').read_text(encoding='utf-8')))
    if project_root:
        project_root = Path(project_root).resolve()
        cfg.update(data_root=str(project_root/'dataset/dataset'),
            manifest=str(project_root/'dataset-audit/local-full/solicited_training_manifest_with_review_flags.csv'))
    for name in ['manifest','data_root']:
        if not Path(cfg[name]).exists(): raise FileNotFoundError(f'{name} tidak ditemukan: {cfg[name]}. Periksa PROJECT_DIR di sel pertama.')
    # Both CPU and GPU machines can execute the classifiers. The notebook's
    # explicit MODE protects local preparation; no artificial GPU requirement.
    cfg.update(package_root=str(package), output_dir=str(package/'runs'/run_name), mode=mode)
    cfg['source_hashes'] = {name:cr.sha256(package/name) for name in
        ['src/coda_research.py','src/coda_v3.py','src/coda_v3_plots.py','config_v3.json']}
    cfg['input_hashes'] = {'manifest':cr.sha256(cfg['manifest']),
        **{p.name:cr.sha256(p) for p in (Path(cfg['data_root'])/'meta_data/Clinical').glob('*.csv')}}
    cfg['protocol_fingerprint'] = digest({k:v for k,v in cfg.items()
        if k not in ['package_root','output_dir','data_root','manifest']})
    out = Path(cfg['output_dir']); out.mkdir(parents=True,exist_ok=True)
    if (out/'resolved_config.json').exists():
        previous = json.loads((out/'resolved_config.json').read_text())
        if cfg['protocol_fingerprint'] != previous['protocol_fingerprint']:
            raise RuntimeError('Run ini memakai kode/data/konfigurasi berbeda. Pertahankan hasil lama dan catat protokol dengan RUN_NAME baru.')
    if (out/'RUN_COMPLETE.json').exists():
        # Run All on a finished protocol reuses folds and rebuilds reports only.
        # Do not overwrite the original environment/creation time.
        checks=json.loads((out/'artifact_checksums.json').read_text())
        if not all((out/p).is_file() and cr.sha256(out/p)==h for p,h in checks.items()):
            raise RuntimeError('Artefak completed run berubah. Audit dahulu; jangan training ulang otomatis.')
        return cfg,json.loads((out/'environment.json').read_text())
    atomic_json(out/'resolved_config.json',cfg)
    if not (out/'RUN_STARTED.json').exists():
        atomic_json(out/'RUN_STARTED.json',{'started_utc':pd.Timestamp.now(tz='UTC').isoformat()})
    cr.seed_everything(cfg['seed'])
    import torch
    torch.set_num_threads(min(cfg['cpu_threads'], os.cpu_count() or 1))
    report = cr.environment_report(cfg)
    atomic_json(out/'candidate_grid.json',candidate_grid(cfg))
    return cfg, report


def audit_cohort(cfg):
    m = pd.read_csv(cfg['manifest'])
    assert m.filename.is_unique and m.tb_status.isin([0,1]).all()
    def check(row):
        path, expected = row
        if cr.sha256(Path(cfg['data_root'])/path) != expected:
            raise ValueError(f'WAV berubah dari manifest: {path}')
    print(f'Verifikasi SHA256 {len(m):,} WAV...',flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i,_ in enumerate(executor.map(check,zip(m.path,m.sha256)),1):
            if i%500==0 or i==len(m):print(f'WAV sesuai: {i:,}/{len(m):,}',flush=True)
    participants,m = cr.prepare_cohort(cfg,verify_hashes=False)
    atomic_json(Path(cfg['output_dir'])/'wav_integrity_audit.json',
        {'n_verified':len(m),'manifest_sha256':cfg['input_hashes']['manifest'],'all_verified':True})
    return participants,m


def nested_splits(participants,cfg):
    table = participants.set_index('participant').sort_index()
    ids = table.index.to_numpy(); y = table.tb_status.to_numpy()
    outer = StratifiedKFold(cfg['outer_folds'],shuffle=True,random_state=cfg['seed'])
    folds = []
    for fold,(train,test) in enumerate(outer.split(ids,y)):
        pool = ids[train]
        proper,calibration = train_test_split(pool,test_size=cfg['outer_train_calibration_fraction'],
            stratify=table.loc[pool,'tb_status'],random_state=cfg['seed']+100+fold)
        proper = np.array(sorted(proper)); calibration = sorted(calibration)
        inner = []
        cv = StratifiedKFold(cfg['inner_folds'],shuffle=True,random_state=cfg['seed']+200+fold)
        for a,b in cv.split(proper,table.loc[proper,'tb_status']):
            inner.append({'train':proper[a].tolist(),'validation':proper[b].tolist()})
        record = {'outer_fold':fold,'proper_train':proper.tolist(),'calibration':calibration,
                  'test':sorted(ids[test]),'inner_folds':inner}
        assert not set(record['proper_train'])&set(calibration)
        assert not (set(record['proper_train'])|set(calibration))&set(record['test'])
        for sp in inner:
            assert set(sp['train'])|set(sp['validation']) == set(record['proper_train'])
            assert not set(sp['train'])&set(sp['validation'])
        folds.append(record)
    all_test = [pid for fold in folds for pid in fold['test']]
    assert len(all_test)==len(ids) and len(set(all_test))==len(ids)
    result = {'folds':folds,'n_participants':len(ids),'unit':'participant',
        'stratification':'TB label only; country is not a predictor',
        'notice':'All eligible local participants; internal exploratory, including participants seen in earlier versions.'}
    path = Path(cfg['output_dir'])/'nested_splits.json'
    if path.exists() and json.loads(path.read_text()) != result:
        raise RuntimeError('Nested split berubah dalam run yang sama.')
    atomic_json(path,result)
    return result


def rich_features(x,sr=16000):
    """47 trajectories x 9 summaries = 423; independent of Sensors' 261 set."""
    win=round(.025*sr); hop=round(.010*sr); nfft=512
    _,_,z = signal.stft(x,fs=sr,nperseg=win,noverlap=win-hop,nfft=nfft,boundary=None,padded=False)
    power=np.abs(z)**2+1e-12; freq=np.linspace(0,sr/2,power.shape[0])
    hz=lambda m:700*(10**(m/2595)-1)
    edges=hz(np.linspace(2595*np.log10(1+50/700),2595*np.log10(1+(sr/2)/700),42))
    bank=np.array([np.maximum(0,np.minimum((freq-edges[i])/(edges[i+1]-edges[i]),
        (edges[i+2]-freq)/(edges[i+2]-edges[i+1]))) for i in range(40)])
    mfcc=dct(np.log(bank@power+1e-12),type=2,axis=0,norm='ortho')[:13]
    delta=np.gradient(mfcc,axis=1); delta2=np.gradient(delta,axis=1)
    total=power.sum(axis=0); centroid=(power*freq[:,None]).sum(axis=0)/total
    bandwidth=np.sqrt((power*(freq[:,None]-centroid)**2).sum(axis=0)/total)
    flatness=np.exp(np.log(power).mean(axis=0))/power.mean(axis=0)
    entropy=-(power/total*np.log(power/total)).sum(axis=0)
    frames=np.lib.stride_tricks.sliding_window_view(x,win)[::hop]
    rms=np.sqrt(np.mean(frames**2,axis=1)+1e-12)
    zcr=np.mean(np.diff(np.signbit(frames),axis=1),axis=1)
    rolloff=freq[np.argmax(np.cumsum(power,axis=0)>=.9*total,axis=0)]
    normalized=power/total
    flux=np.r_[0,np.mean(np.maximum(0,np.diff(normalized,axis=1))**2,axis=0)]
    trajectories=np.vstack([mfcc,delta,delta2,centroid,bandwidth,flatness,entropy,rms,zcr,rolloff,flux])
    names=[f'{group}{i}' for group in ['mfcc','delta','delta2'] for i in range(13)]+[
        'centroid','bandwidth','flatness','entropy','rms','zcr','rolloff90','positive_flux']
    centered=trajectories-trajectories.mean(axis=1,keepdims=True)
    sd=trajectories.std(axis=1); safe=np.maximum(sd,1e-8)
    statistics=[trajectories.mean(axis=1),sd,np.mean(centered**3,axis=1)/safe**3,
                np.mean(centered**4,axis=1)/safe**4-3,
                *[np.percentile(trajectories,p,axis=1) for p in [10,25,50,75,90]]]
    values=np.nan_to_num(np.concatenate(statistics),nan=0.,posinf=0.,neginf=0.)
    columns=[f'{stat}_{name}' for stat in ['mean','std','skew','kurt','p10','p25','p50','p75','p90'] for name in names]
    assert len(values)==423 and np.isfinite(values).all()
    return values,columns


def prepare_features(participants,manifest,cfg):
    fingerprint=digest({'inputs':cfg['input_hashes'],'code':cfg['source_hashes']['src/coda_v3.py'],
        'base_code':cfg['source_hashes']['src/coda_research.py'],'sr':16000,'seconds':.5})
    folder=Path(cfg['package_root'])/'cache_v3'/fingerprint[:16]; folder.mkdir(parents=True,exist_ok=True)
    final=folder/'participant_features.joblib'; meta_path=folder/'complete.json'
    if final.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['fingerprint']!=fingerprint or cr.sha256(final)!=meta['sha256']:
            raise RuntimeError('Cache fitur berubah atau tidak lengkap.')
        table=joblib.load(final)
        assert set(table.index)==set(participants.participant)
        print('Memakai cache fitur lengkap:',folder,flush=True)
    else:
        ordered=manifest.sort_values('filename').reset_index(drop=True)
        basic=[]; rich=[]; start=time.perf_counter()
        for i,row in ordered.iterrows():
            x=cr.read_audio(Path(cfg['data_root'])/row.path,cfg)
            f,fn=cr.acoustic_features(x,16000); r,rn=rich_features(x)
            basic.append(f); rich.append(r)
            if (i+1)%500==0 or i+1==len(ordered):
                print(f'Fitur: {i+1:,}/{len(ordered):,} batuk | {time.perf_counter()-start:.0f} detik',flush=True)
        b=pd.DataFrame(basic,columns=fn); b['participant']=ordered.participant.values
        r=pd.DataFrame(rich,columns=rn); r['participant']=ordered.participant.values
        pooled=[b.groupby('participant').mean().add_prefix('mfcc_'),
                r.groupby('participant').mean().add_prefix('rich_mean_'),
                r.groupby('participant').std(ddof=0).add_prefix('rich_std_')]
        table=participants.set_index('participant').join(pooled).sort_index()
        atomic_model(final,table)
        atomic_json(meta_path,{'fingerprint':fingerprint,'sha256':cr.sha256(final),
            'n_audio':len(ordered),'n_participants':len(table),'seconds':time.perf_counter()-start,
            'method':'MFCC 86 / rich 423 per clip; patient mean or mean+population std; per-clip z-score, no learned cohort statistics'})
    # Include features in RETURN, so CPU audit does not require another audio pass.
    target=Path(cfg['output_dir'])/'participant_features.joblib'
    if not target.exists() or cr.sha256(target)!=cr.sha256(final):
        import shutil
        shutil.copyfile(final,target)
    atomic_json(Path(cfg['output_dir'])/'feature_provenance.json',json.loads(meta_path.read_text()))
    return table,folder


def groups_for(table,candidate):
    groups={}
    for g in COMBINATIONS[candidate['combination']]:
        if g=='clinical':cols=CLINICAL
        elif g=='hr':cols=['heart_rate']
        else:
            prefix={'mfcc_mean':('mfcc_audio_',),'rich_mean':('rich_mean_',),
                'rich_mean_std':('rich_mean_','rich_std_')}[candidate['representation']]
            cols=[c for c in table if c.startswith(prefix)]
        if not cols:raise ValueError(f'Fitur tidak tersedia: {candidate}')
        groups[g]=list(cols)
    forbidden={'tb_status','Country','participant','height','weight','temperature','cough_count','sound_prediction_score'}
    assert not set(itertools.chain.from_iterable(groups.values()))&forbidden
    return groups


def candidate_grid(cfg):
    candidates=[]
    for combo,groups in COMBINATIONS.items():
        for rep in cfg['representations'] if 'audio' in groups else ['none']:
            for c in cfg['lr_C']:
                candidates.append({'combination':combo,'representation':rep,'family':'lr','fusion':'early','C':c})
            for parameters in cfg['catboost_trials']:
                candidates.append({'combination':combo,'representation':rep,'family':'catboost',
                    'fusion':'early','params':parameters})
            if 'audio' in groups and len(groups)>1:
                for weight in cfg['late_fusion_weights']:
                    candidates.append({'combination':combo,'representation':rep,'family':'lr','fusion':'late',
                        'C':cfg['late_fusion_C'],'audio_weight':weight})
    for i,c in enumerate(candidates):c['id']=f'c{i:03d}'
    return candidates


def _preprocessor(table,ids,cols,audio,cfg):
    key=(id(table),tuple(ids),tuple(cols),cfg['pca_components'],cfg['seed'])
    if key in _PP_CACHE:return _PP_CACHE[key]
    if audio:
        pp=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
            ('scale',StandardScaler()),('pca',PCA(n_components=min(cfg['pca_components'],len(ids)-1,len(cols)),
                svd_solver='randomized',random_state=cfg['seed']))])
    else:pp=cr.make_preprocessor(cols)
    pp.fit(table.loc[ids,cols]);_PP_CACHE[key]=pp
    return pp


def fit_model(table,ids,candidate,cfg,seed):
    groups=groups_for(table,candidate)
    if candidate['fusion']=='late':
        a=copy.deepcopy(candidate);a.update(combination='audio',fusion='early')
        t=copy.deepcopy(candidate);t.update(combination='clinical_hr' if len(groups)==3 else
            'clinical' if 'clinical' in groups else 'hr',representation='none',fusion='early')
        return {'candidate':candidate,'groups':groups,'fit_ids':list(ids),
            'audio':fit_model(table,ids,a,cfg,seed),'tab':fit_model(table,ids,t,cfg,seed)}
    blocks=[(g,cols,_preprocessor(table,ids,cols,g=='audio',cfg)) for g,cols in groups.items()]
    x=np.concatenate([p.transform(table.loc[ids,cols]) for _,cols,p in blocks],axis=1)
    if candidate['family']=='lr':
        model=LogisticRegression(C=candidate['C'],class_weight='balanced',max_iter=5000,random_state=seed)
    else:
        from catboost import CatBoostClassifier
        model=CatBoostClassifier(**candidate['params'],auto_class_weights='Balanced',random_seed=seed,
            task_type='CPU',thread_count=cfg['cpu_threads'],verbose=False,allow_writing_files=False)
    model.fit(x,table.loc[ids,'tb_status'].astype(int))
    return {'candidate':candidate,'groups':groups,'blocks':blocks,'model':model,'fit_ids':list(ids)}


def predict_raw(model,table):
    if model['candidate']['fusion']=='late':
        weight=model['candidate']['audio_weight']
        return weight*predict_raw(model['audio'],table)+(1-weight)*predict_raw(model['tab'],table)
    x=np.concatenate([pp.transform(table[cols]) for _,cols,pp in model['blocks']],axis=1)
    return model['model'].predict_proba(x)[:,1]


def predict(bundle,table):
    return np.clip(bundle['calibrator'].predict(predict_raw(bundle['model'],table)),0,1)


def threshold(y,p,policy,target=.9):
    y=np.asarray(y,int);p=np.asarray(p,float)
    fpr,tpr,cuts=roc_curve(y,p,drop_intermediate=False)
    cuts[~np.isfinite(cuts)]=np.nextafter(float(p.max()),np.inf)
    if policy=='youden':
        j=tpr-fpr;best=sorted(range(len(cuts)),key=lambda i:(-j[i],fpr[i],-cuts[i]))[0]
    elif policy=='sensitivity90':
        eligible=np.flatnonzero(tpr>=target)
        best=sorted(eligible,key=lambda i:(fpr[i],-cuts[i]))[0]
    else:raise ValueError(policy)
    return float(cuts[best])


def calculate_metrics(y,p,cut):
    m=cr.metrics(y,p,cut)
    m.update(uar=(m['sensitivity']+m['specificity'])/2,
             referral_fraction=(m['tp']+m['fp'])/m['n'])
    return m


def inner_search(table,fold,cfg):
    folder=Path(cfg['output_dir'])/f'outer_{fold["outer_fold"]:02d}'
    signature=digest({'protocol':cfg['protocol_fingerprint'],'fold':fold})
    records=[];scores=[];grid=candidate_grid(cfg)
    for i,candidate in enumerate(grid):
        path=folder/'inner_checkpoints'/(candidate['id']+'.json')
        if path.exists():
            cached=json.loads(path.read_text())
            if cached['signature']!=signature or cached['candidate']!=candidate:raise RuntimeError('Inner checkpoint mismatch')
            predictions=cached['predictions'];stats=cached['statistics']
        else:
            start=time.perf_counter();predictions=[];stats=[]
            for inner,sp in enumerate(fold['inner_folds']):
                inner_path=folder/'inner_checkpoints'/f'{candidate["id"]}_inner{inner}.json'
                if inner_path.exists():
                    saved=json.loads(inner_path.read_text())
                    if saved['signature']!=signature or saved['candidate']!=candidate:raise RuntimeError('Partial inner fold mismatch')
                    stats.append(saved['statistics']);predictions.extend(saved['predictions'])
                    continue
                model=fit_model(table,sp['train'],candidate,cfg,cfg['seed']+1000*fold['outer_fold']+inner)
                p=predict_raw(model,table.loc[sp['validation']]);y=table.loc[sp['validation'],'tb_status'].to_numpy()
                # Mirrors Sensors: inner-validation Youden used for ranking only.
                # Never reuse this cutoff for outer testing or deployment.
                cut=threshold(y,p,'youden');m=calculate_metrics(y,p,cut)
                statistic={'inner_fold':inner,'auc':m['roc_auc'],'uar':m['uar'],'inner_ranking_cutoff':cut}
                prediction=[{'participant':pid,'y':int(yy),'raw_probability':float(pp),'inner_fold':inner}
                    for pid,yy,pp in zip(sp['validation'],y,p)]
                atomic_json(inner_path,{'signature':signature,'candidate':candidate,'statistics':statistic,'predictions':prediction})
                stats.append(statistic);predictions.extend(prediction)
            atomic_json(path,{'signature':signature,'candidate':candidate,'predictions':predictions,
                'statistics':stats,'fit_seconds':time.perf_counter()-start})
        assert len(predictions)==len(fold['proper_train']) and len({r['participant'] for r in predictions})==len(predictions)
        scores.append({**candidate,'mean_inner_uar':float(np.mean([r['uar'] for r in stats])),
            'mean_inner_auc':float(np.mean([r['auc'] for r in stats])),
            'std_inner_auc':float(np.std([r['auc'] for r in stats],ddof=1))})
        records.extend({'candidate':candidate['id'],**r} for r in predictions)
        if (i+1)%5==0 or i+1==len(grid):
            print(f'Outer {fold["outer_fold"]+1}/{cfg["outer_folds"]} | kandidat {i+1}/{len(grid)}',flush=True)
        pd.DataFrame(scores).to_csv(folder/'inner_search.csv',index=False)
    scores=pd.DataFrame(scores);oof=pd.DataFrame(records)
    oof.to_csv(folder/'proper_oof_all_candidates.csv',index=False)
    winners={combo:str(scores[scores.combination==combo].sort_values(
        ['mean_inner_uar','mean_inner_auc','id'],ascending=[False,False,True]).iloc[0]['id']) for combo in COMBINATIONS}
    atomic_json(folder/'selection.json',{'winners':winners,'rule':cfg['selection_rule'],
        'notice':'No outer calibration or outer test labels used to select model.'})
    return scores,oof,winners


def finalize_fold(table,fold,oof,winners,cfg):
    folder=Path(cfg['output_dir'])/f'outer_{fold["outer_fold"]:02d}'
    grid={c['id']:c for c in candidate_grid(cfg)};bundles={}
    for combo,cid in winners.items():
        path=folder/'models'/combo/'bundle.joblib'
        if path.exists():
            b=joblib.load(path)
            if b['protocol_fingerprint']!=cfg['protocol_fingerprint'] or b['candidate']!=grid[cid]:raise RuntimeError('Saved model mismatch')
        else:
            selected=oof[oof.candidate==cid].sort_values('participant')
            assert set(selected.participant)==set(fold['proper_train'])
            iso=IsotonicRegression(y_min=0,y_max=1,increasing=True,out_of_bounds='clip')
            iso.fit(selected.raw_probability,selected.y)
            model=fit_model(table,fold['proper_train'],grid[cid],cfg,cfg['seed']+1000*fold['outer_fold']+99)
            b={'model':model,'calibrator':iso,'candidate':grid[cid],'groups':model['groups'],
                'outer_fold':fold['outer_fold'],'protocol_fingerprint':cfg['protocol_fingerprint'],
                'proper_ids':fold['proper_train'],'calibration_ids':fold['calibration']}
            cal=table.loc[fold['calibration']];p=predict(b,cal)
            b['thresholds']={policy:threshold(cal.tb_status,p,policy,cfg['target_sensitivity']) for policy in POLICIES}
            atomic_model(path,b)
            atomic_json(path.parent/'metadata.json',{k:v for k,v in b.items() if k not in ['model','calibrator']})
            pd.DataFrame({'participant':cal.index,'y':cal.tb_status,'probability':p,
                'raw_probability':predict_raw(model,cal)}).to_csv(path.parent/'calibration_predictions.csv',index=False)
        bundles[combo]=b
    atomic_json(folder/'MODELS_AND_THRESHOLDS_FROZEN.json',{combo:{'candidate':b['candidate'],
        'thresholds':b['thresholds'],'bundle_sha256':cr.sha256(folder/'models'/combo/'bundle.joblib')} for combo,b in bundles.items()})
    return bundles


def evaluate_fold(bundles,table,fold,cfg):
    folder=Path(cfg['output_dir'])/f'outer_{fold["outer_fold"]:02d}'
    assert (folder/'MODELS_AND_THRESHOLDS_FROZEN.json').exists()
    test=table.loc[fold['test']];predictions=[];metrics=[];parity=[]
    for combo,bundle in bundles.items():
        p=predict(bundle,test);raw=predict_raw(bundle['model'],test)
        loaded=joblib.load(folder/'models'/combo/'bundle.joblib')
        error=float(np.max(np.abs(p-predict(loaded,test))))
        assert error<1e-10
        parity.append({'combination':combo,'max_cpu_reload_error':error})
        for policy,cut in bundle['thresholds'].items():
            metrics.append({'outer_fold':fold['outer_fold'],'combination':combo,'policy':policy,
                'raw_roc_auc':roc_auc_score(test.tb_status,raw),**calculate_metrics(test.tb_status,p,cut)})
            predictions.extend({'outer_fold':fold['outer_fold'],'combination':combo,'policy':policy,
                'participant':pid,'y':int(yy),'Country':country,'probability':float(pp),'raw_probability':float(rr),
                'threshold':cut,'prediction':int(pp>=cut)} for pid,yy,country,pp,rr in
                zip(test.index,test.tb_status,test.Country,p,raw))
    pd.DataFrame(predictions).to_csv(folder/'outer_test_predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(folder/'outer_test_metrics.csv',index=False)
    atomic_json(folder/'inference_parity.json',parity)
    return pd.DataFrame(metrics),pd.DataFrame(predictions)


def _shapley(bundle,table,pid,background):
    keys=list(bundle['groups']);n=len(keys);coalitions=list(itertools.product((0,1),repeat=n));frames=[]
    for bits in coalitions:
        t=background.copy()
        for present,group in zip(bits,keys):
            if present:
                # Scalar assignment preserves numeric/categorical column dtypes,
                # including the case of an entirely missing HR coalition.
                for col in bundle['groups'][group]:t[col]=table.at[pid,col]
        frames.append(t)
    values=dict(zip(coalitions,predict(bundle,pd.concat(frames)).reshape(len(frames),len(background)).mean(axis=1)))
    phi={}
    for j,group in enumerate(keys):
        phi[group]=0.
        for bits,value in values.items():
            if bits[j]:continue
            k=sum(bits);other=list(bits);other[j]=1
            phi[group]+=math.factorial(k)*math.factorial(n-k-1)/math.factorial(n)*(values[tuple(other)]-value)
    base=float(values[(0,)*n]);probability=float(predict(bundle,table.loc[[pid]])[0])
    residual=probability-base-sum(phi.values())
    assert abs(residual)<1e-8
    return {'base_probability':base,'probability':probability,'completeness_residual':residual,**phi}


def explain_fold(bundles,table,fold,cfg):
    folder=Path(cfg['output_dir'])/f'outer_{fold["outer_fold"]:02d}'
    rng=np.random.default_rng(cfg['seed']+3000+fold['outer_fold'])
    # Fix IDs independently of errors/scores; background never contains test.
    test_ids=sorted(rng.choice(fold['test'],min(cfg['xai_per_outer_fold'],len(fold['test'])),replace=False))
    backgrounds=[list(rng.choice(fold['proper_train'],min(cfg['xai_background'],len(fold['proper_train'])),replace=False)) for _ in range(2)]
    rows=[]
    for combo,bundle in bundles.items():
        for pid in test_ids:
            primary=_shapley(bundle,table,pid,table.loc[backgrounds[0]])
            secondary=_shapley(bundle,table,pid,table.loc[backgrounds[1]])
            rows.append({'outer_fold':fold['outer_fold'],'combination':combo,'participant':pid,**primary,
                **{group+'_background_abs_delta':abs(primary[group]-secondary[group]) for group in bundle['groups']}})
    result=pd.DataFrame(rows);result.to_csv(folder/'grouped_xai.csv',index=False)
    atomic_json(folder/'xai_definition.json',{'test_ids':test_ids,'background_id_sets':backgrounds,
        'method':'Exact grouped interventional Shapley of fold model including preprocessing/fusion/isotonic',
        'warning':'Not causal; group substitution breaks correlation. Small fixed samples, not population feature importance.'})
    return result


def robustness_fold(bundles,table,fold,cfg):
    test=table.loc[fold['test']];rows=[];rng=np.random.default_rng(cfg['seed']+4000+fold['outer_fold'])
    for combo,bundle in bundles.items():
        original=predict(bundle,test);conditions={'original':test.copy()}
        if 'hr' in bundle['groups']:
            for shift in cfg['hr_shifts_bpm']:
                t=test.copy();t['heart_rate']+=shift;t.loc[~t.heart_rate.between(30,220),'heart_rate']=np.nan
                conditions[f'hr_shift_{shift:+d}']=t
            t=test.copy();t['heart_rate']=np.nan;conditions['hr_missing_primary_diagnostic']=t
        if 'audio' in bundle['groups']:
            t=test.copy();cols=bundle['groups']['audio'];t[cols]=t[cols].to_numpy()[rng.permutation(len(t))]
            conditions['audio_permuted']=t
        for condition,altered in conditions.items():
            p=predict(bundle,altered)
            for policy,cut in bundle['thresholds'].items():
                rows.append({'outer_fold':fold['outer_fold'],'combination':combo,'condition':condition,'policy':policy,
                    **calculate_metrics(test.tb_status,p,cut),'mean_abs_probability_change':float(np.abs(p-original).mean()),
                    'changed_decisions':int(np.sum((p>=cut)!=(original>=cut)))})
    result=pd.DataFrame(rows)
    result.to_csv(Path(cfg['output_dir'])/f'outer_{fold["outer_fold"]:02d}'/'robustness.csv',index=False)
    return result


def run_nested_cv(table,splits,cfg,on_fold=None):
    out=Path(cfg['output_dir'])
    for fold in splits['folds']:
        folder=out/f'outer_{fold["outer_fold"]:02d}'
        if (folder/'FOLD_COMPLETE.json').exists():
            complete=json.loads((folder/'FOLD_COMPLETE.json').read_text())
            if complete['protocol_fingerprint']!=cfg['protocol_fingerprint']:raise RuntimeError('Completed fold fingerprint mismatch')
            assert all(cr.sha256(folder/p)==h for p,h in complete['checksums'].items())
            print(f'Outer {fold["outer_fold"]+1} lengkap, memakai hasil tersimpan.',flush=True)
            if on_fold:on_fold(fold)
            continue
        print(f'\n=== OUTER {fold["outer_fold"]+1}/{len(splits["folds"])} ===',flush=True)
        _PP_CACHE.clear();scores,oof,winners=inner_search(table,fold,cfg)
        bundles=finalize_fold(table,fold,oof,winners,cfg)
        evaluate_fold(bundles,table,fold,cfg)
        explain_fold(bundles,table,fold,cfg)
        robustness_fold(bundles,table,fold,cfg)
        checks={p.relative_to(folder).as_posix():cr.sha256(p) for p in folder.rglob('*')
            if p.is_file() and p.name!='FOLD_COMPLETE.json' and not p.name.endswith('.part')}
        atomic_json(folder/'FOLD_COMPLETE.json',{'protocol_fingerprint':cfg['protocol_fingerprint'],
            'completed_utc':pd.Timestamp.now(tz='UTC').isoformat(),'checksums':checks})
        _PP_CACHE.clear()
        print('Outer selesai; model, cutoff, evaluasi, XAI dan checksums tersimpan.',flush=True)
        if on_fold:on_fold(fold)


def count_interval(k,n):
    return (float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.)


def aggregate_results(splits,cfg):
    out=Path(cfg['output_dir'])
    def concatenate(name):
        return pd.concat([pd.read_csv(out/f'outer_{f["outer_fold"]:02d}'/name,float_precision='round_trip')
            for f in splits['folds']],ignore_index=True)
    metrics=concatenate('outer_test_metrics.csv');predictions=concatenate('outer_test_predictions.csv')
    xai=concatenate('grouped_xai.csv');robust=concatenate('robustness.csv')
    for (combo,policy),df in predictions.groupby(['combination','policy']):
        assert len(df)==splits['n_participants'] and df.participant.is_unique
    metric_names=['roc_auc','raw_roc_auc','pr_auc','sensitivity','specificity','ppv','npv','uar','brier','referral_fraction']
    summary=metrics.groupby(['combination','policy'])[metric_names].agg(['mean','std','count']).reset_index()
    summary.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in summary.columns]
    pooled=[]
    for (combo,policy),df in predictions.groupby(['combination','policy']):
        y=df.y.to_numpy();p=df.probability.to_numpy();pred=df.prediction.to_numpy()
        # Different fold cutoffs: count saved decisions, never select pooled cutoff.
        tp=int(np.sum((y==1)&(pred==1)));fn=int(np.sum((y==1)&(pred==0)))
        tn=int(np.sum((y==0)&(pred==0)));fp=int(np.sum((y==0)&(pred==1)))
        slo,shi=count_interval(tp,tp+fn);tlo,thi=count_interval(tn,tn+fp)
        pooled.append({'combination':combo,'policy':policy,'n':len(y),'roc_auc':roc_auc_score(y,p),
            'pr_auc':average_precision_score(y,p),'tp':tp,'fn':fn,'tn':tn,'fp':fp,
            'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'ppv':tp/(tp+fp) if tp+fp else np.nan,
            'npv':tn/(tn+fn) if tn+fn else np.nan,'sens_count_ci_low':slo,'sens_count_ci_high':shi,
            'spec_count_ci_low':tlo,'spec_count_ci_high':thi,'referral_fraction':float(pred.mean())})
    pooled=pd.DataFrame(pooled)
    paired=[];rng=np.random.default_rng(cfg['seed'])
    comparisons=[('audio_hr','audio'),('audio_hr','hr'),('audio_clinical_hr','clinical_hr'),
                 ('audio_clinical','clinical'),('audio_clinical_hr','audio_clinical')]
    # AUROC does not depend on cutoff: use one policy, not duplicate participants.
    unique=predictions[predictions.policy=='youden']
    for a,b in comparisons:
        aa=unique[unique.combination==a].sort_values('participant');bb=unique[unique.combination==b].sort_values('participant')
        assert aa.participant.tolist()==bb.participant.tolist() and np.array_equal(aa.y,bb.y)
        y=aa.y.to_numpy();pa=aa.probability.to_numpy();pb=bb.probability.to_numpy();boot=[]
        for _ in range(cfg['bootstrap_repeats']):
            ix=np.concatenate([rng.choice(np.flatnonzero(y==k),int((y==k).sum()),replace=True) for k in (0,1)])
            boot.append(roc_auc_score(y[ix],pa[ix])-roc_auc_score(y[ix],pb[ix]))
        fold_a=metrics[(metrics.combination==a)&(metrics.policy=='youden')].set_index('outer_fold').roc_auc
        fold_b=metrics[(metrics.combination==b)&(metrics.policy=='youden')].set_index('outer_fold').roc_auc
        differences=fold_a-fold_b
        paired.append({'comparison':a+' minus '+b,'pooled_auc_delta':roc_auc_score(y,pa)-roc_auc_score(y,pb),
            'conditional_bootstrap_low':np.quantile(boot,.025),'conditional_bootstrap_high':np.quantile(boot,.975),
            'mean_fold_auc_delta':differences.mean(),'std_fold_auc_delta':differences.std(ddof=1),
            'outer_folds_positive_delta':int((differences>0).sum()),'n_outer_folds':len(differences)})
    paired=pd.DataFrame(paired)
    for name,df in [('nested_fold_metrics.csv',metrics),('nested_oof_predictions.csv',predictions),
        ('nested_mean_std.csv',summary),('pooled_oof_metrics.csv',pooled),('paired_modality_comparisons.csv',paired),
        ('nested_grouped_xai.csv',xai),('nested_robustness.csv',robust)]:df.to_csv(out/name,index=False)
    atomic_json(out/'statistical_interpretation.json',{
        'primary_report':'Mean +/- sample SD over outer folds, not a confidence interval',
        'pooled_oof':'Secondary; calibrated score distributions can differ across folds; no common tuned threshold',
        'count_intervals':'Descriptive binomial count intervals; overlapping CV training sets mean these do not capture full retraining/population uncertainty',
        'paired_bootstrap':'Exploratory conditional participant bootstrap; no multiplicity correction, no retraining uncertainty',
        'prior_exposure':'Cohort used in V1/V2; nested CV here is internal exploratory, not independent final validation'})
    return metrics,predictions,summary,pooled,paired,xai,robust


def export_return(cfg):
    out=Path(cfg['output_dir']);package=Path(cfg['package_root'])
    if not (out/'RUN_COMPLETE.json').exists():
        atomic_json(out/'RUN_COMPLETE.json',{'completed_utc':pd.Timestamp.now(tz='UTC').isoformat(),
            'protocol_fingerprint':cfg['protocol_fingerprint'],'internal_exploratory':True,
            'camera_hr_reference_validated':False,'deployment_model_selected':False})
    checks={p.relative_to(out).as_posix():cr.sha256(p) for p in out.rglob('*')
        if p.is_file() and p.name!='artifact_checksums.json' and not p.name.endswith('.part')}
    atomic_json(out/'artifact_checksums.json',checks)
    target=out.with_name(out.name+'_RETURN.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in out.rglob('*'):
            if p.is_file() and not p.name.endswith('.part'):z.write(p,'run/'+p.relative_to(out).as_posix())
        for name in ['src/__init__.py','src/coda_research.py','src/coda_v3.py','src/coda_v3_plots.py',
            'config.json','config_v3.json','CODA_TB_V3_NestedCV.ipynb','V3_README.md']:
            z.write(package/name,'research_package/'+name)
    return target
