"""V5 fixed-budget models; no outer-test data enter fit or selection."""
from pathlib import Path
import gc
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from . import coda_research as cr

CLINICAL = ['age','sex','height','weight','reported_cough_dur','tb_prior',
            'tb_prior_Pul','tb_prior_Extrapul','tb_prior_Unknown','hemoptysis',
            'temperature','weight_loss','smoke_lweek','fever','night_sweats']
MOBILE = ['age','sex','reported_cough_dur','tb_prior','hemoptysis',
          'weight_loss','smoke_lweek','fever','night_sweats']
OUTPUTS = ['clinical_full','paper_full_lr','paper_full_cb','paper_full_tabpfn',
           'paper_pooled_lr','paper_pooled_cb',
           'beats_full_lr','beats_full_cb','beats_full_tabpfn','ensemble_full',
           'beats_hr','beats_mobile']
LABELS = {k:k.replace('_',' ') for k in OUTPUTS}


def candidate_grid(cfg):
    grid=[]
    def add(output,rep,family,trial=0,pca=None):
        c=dict(id=f'{output}_{sum(x["output"]==output for x in grid):02d}',
               output=output,representation=rep,family=family,pca=pca)
        if family=='lr': c['C']=cfg['lr_C'][trial]
        elif family=='cb': c['params']=cfg['catboost_trials'][trial]
        elif family=='tabpfn': c['n_estimators']=cfg['tabpfn_estimators']
        grid.append(c);return c['id']
    for family in ['lr','cb','tabpfn']:
        for j in range(2):add('clinical_full','none',family,j)
    # TabPFN duplicates on clinical input differ by deterministic ensemble seed.
    for j,c in enumerate(grid): c['seed_offset']=j if c['family']=='tabpfn' else 0
    for rep in ['paper','beats']:
        for family in ['lr','cb']:
            for j in range(3):add(f'{rep}_full_{family}',rep,family,j,64 if rep=='beats' else None)
        for pca in cfg['pca_components']:
            add(f'{rep}_full_tabpfn',rep,'tabpfn',pca=pca)
    for output in ['beats_hr','beats_mobile']:
        for family in ['lr','cb','tabpfn']:
            for j in range(2):add(output,'beats',family,j,cfg['pca_components'][j])
    # Matched pooling/head/PCA controls isolate representation from aggregation.
    for family in ['lr','cb']:
        for j in range(3):
            add('paper_pooled_'+family,'paper',family,j,64)
            grid[-1]['pooling']='patient'
    # Fixed pairs/weights, not bases selected using in-sample meta predictions.
    for weight in [.25,.5,.75]:
        grid.append(dict(id=f'ensemble_full_{len([c for c in grid if c["output"]=="ensemble_full"]):02d}',
            output='ensemble_full',representation='beats',family='blend',
            components=['beats_full_cb_01','clinical_full_03'],weights=[weight,1-weight]))
    grid.append(dict(id='ensemble_full_03',output='ensemble_full',representation='beats+paper',family='blend',
        components=['beats_full_cb_01','paper_full_cb_01'],weights=[.5,.5]))
    assert len(grid)==44 and len({c['id'] for c in grid})==44
    assert set(c['output'] for c in grid)==set(OUTPUTS)
    return grid


def groups_for(table,candidate):
    output=candidate['output'];groups={}
    rep=candidate['representation']
    if rep!='none':
        groups['audio']=['_bag_'+r for r in rep.split('+')]
    if output!='beats_hr':groups['clinical']=MOBILE if output=='beats_mobile' else CLINICAL
    groups['hr']=['heart_rate']
    assert all(c in table for cols in groups.values() for c in cols)
    assert not set(sum(groups.values(),[])) & {'tb_status','Country','participant','HIVstatus',
        'Microbiologicreferencestandard','Sputumxpertreferencestandard','Xpertcombinedsemiquant','cough_count'}
    return groups


def audio_preprocessor(x,pca,seed):
    steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]
    if pca:steps.append(('pca',PCA(n_components=min(pca,x.shape[1],len(x)-1),svd_solver='randomized',random_state=seed)))
    return Pipeline(steps).fit(x)


def pooled_audio(t,col):
    # Fixed, order-invariant pooling. Number of coughs is not a predictor.
    return np.stack([np.concatenate([b.mean(axis=0,dtype=np.float64),b.std(axis=0,dtype=np.float64)]) for b in t[col]])


def fit_model(table,ids,candidate,cfg,seed):
    t=table.loc[list(ids)];y=t.tb_status.to_numpy(dtype=int)
    assert set(np.unique(y))=={0,1}
    groups=groups_for(table,candidate)
    model=dict(candidate=candidate,groups=groups,fit_ids=list(ids),seed=seed)
    if candidate['family']=='blend':
        grid={c['id']:c for c in candidate_grid(cfg)}
        model['components']=[fit_model(table,ids,grid[c],cfg,seed) for c in candidate['components']]
        return model
    tabcols=sum([cols for g,cols in groups.items() if g!='audio'],[])
    # All clinical fields are deterministically numeric encoded before this fit.
    pp=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]).fit(t[tabcols])
    tab=pp.transform(t[tabcols]);model.update(tab_columns=tabcols,tab_preprocessor=pp)
    family=candidate['family'];rep=candidate['representation']
    clip=(rep=='paper' and family!='tabpfn' and candidate.get('pooling')!='patient')
    model['clip']=clip
    if rep!='none':
        col=groups['audio'][0];bags=t[col].tolist();counts=np.array([len(b) for b in bags])
        audio=np.concatenate(bags).astype(np.float64) if clip else pooled_audio(t,col)
        ap=audio_preprocessor(audio,candidate.get('pca'),seed);model['audio_preprocessor']=ap
        x=np.column_stack([ap.transform(audio),np.repeat(tab,counts,axis=0) if clip else tab])
    else:x=tab
    target=np.repeat(y,counts) if clip else y
    if family=='lr':classifier=LogisticRegression(C=candidate['C'],max_iter=10000,random_state=seed)
    elif family=='cb':
        from catboost import CatBoostClassifier
        classifier=CatBoostClassifier(**candidate['params'],random_seed=seed,thread_count=cfg['cpu_threads'],
            verbose=False,allow_writing_files=False,task_type='CPU')
    elif family=='tabpfn':
        from tabpfn import TabPFNClassifier
        from .coda_v5_features import gpu_guard
        gpu_guard(cfg,'before TabPFN')
        classifier=TabPFNClassifier(n_estimators=candidate['n_estimators'],device=cfg['tabpfn_device'],
            model_path=str(Path(cfg['package_root'])/cfg['tabpfn_checkpoint']),
            random_state=seed+candidate.get('seed_offset',0),fit_mode=cfg['tabpfn_fit_mode'],
            memory_saving_mode=True,n_jobs=cfg['cpu_threads'])
        model['tabpfn_data']=(x.astype(np.float32),target.copy())
        # Retain training context, not a CUDA object, for portable checkpointing.
        classifier.fit(*model['tabpfn_data'])
        model['model']=classifier
        return model
    else:raise ValueError(family)
    classifier.fit(x,target)
    model['model']=classifier
    return model


def predict_raw(model,table,cfg=None):
    c=model['candidate']
    if c['family']=='blend':
        return sum(w*predict_raw(m,table,cfg) for w,m in zip(c['weights'],model['components']))
    tab=model['tab_preprocessor'].transform(table[model['tab_columns']])
    if c['representation']!='none':
        col=model['groups']['audio'][0]
        if model['clip']:
            bags=table[col].tolist();counts=np.array([len(b) for b in bags]);cuts=np.r_[0,np.cumsum(counts)]
            audio=np.concatenate(bags).astype(np.float64);tab=np.repeat(tab,counts,axis=0)
        else:audio=pooled_audio(table,col)
        x=np.column_stack([model['audio_preprocessor'].transform(audio),tab])
    else:x=tab
    if c['family']=='tabpfn':
        if 'model' not in model:
            from tabpfn import TabPFNClassifier
            path=Path(cfg['package_root'])/cfg['tabpfn_checkpoint']
            model['model']=TabPFNClassifier(n_estimators=c['n_estimators'],device=cfg['tabpfn_device'],
                model_path=str(path),random_state=model['seed']+c.get('seed_offset',0),fit_mode=cfg['tabpfn_fit_mode'],
                memory_saving_mode=True,n_jobs=cfg['cpu_threads']).fit(*model['tabpfn_data'])
        x=x.astype(np.float32)
        chunk=cfg.get('tabpfn_prediction_chunk',128)
        p=np.concatenate([model['model'].predict_proba(x[i:i+chunk])[:,1] for i in range(0,len(x),chunk)])
    else:p=model['model'].predict_proba(x)[:,1]
    if model['clip']:p=np.array([p[cuts[j]:cuts[j+1]].mean() for j in range(len(table))])
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid model probabilities')
    return np.asarray(p,dtype=float)


def release(model):
    if model['candidate']['family']=='blend':
        for m in model['components']:release(m)
        return
    if model['candidate']['family']!='tabpfn':return
    model.pop('model',None)
    gc.collect()
    import torch
    if torch.cuda.is_available():torch.cuda.empty_cache()
