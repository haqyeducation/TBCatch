"""CODA subset experiments. All learned preprocessing stays inside development folds.

This is an independent research implementation, not the winning team's code.
The mobile model uses no measured temperature, anthropometrics, country or lab results.
"""
from __future__ import annotations
import copy, hashlib, io, json, math, os, platform, random, subprocess, time, wave, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import dct
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, log_loss, roc_curve)
import joblib

NUMERIC = {'age','height','weight','reported_cough_dur','heart_rate','temperature'}
VERSION = '1.0.0'

def save_json(path, obj):
    def clean(x):
        if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not math.isfinite(x): return None
        if isinstance(x, Path): return str(x)
        return x
    path=Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj),indent=2,ensure_ascii=False),encoding='utf-8')

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def load_config(package_root, config_name='config.json'):
    root=Path(package_root).resolve();cfg=json.loads((root/config_name).read_text())
    for k in ['data_root','manifest','output_dir']:
        cfg[k]=str((root/cfg[k]).resolve())
    cfg['package_root']=str(root)
    return cfg

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True, warn_only=True)

def environment_report(cfg):
    import importlib.metadata as md
    packages={}
    for name in ['numpy','pandas','scipy','scikit-learn','catboost','torch','matplotlib','nbformat','nbclient','joblib']:
        try:packages[name]=md.version(name)
        except md.PackageNotFoundError:packages[name]=None
    import torch
    report={'python':platform.python_version(),'platform':platform.platform(),'packages':packages,
        'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'gpu_bytes':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        'code_sha256':sha256(Path(__file__)),'config':cfg,'created_utc':pd.Timestamp.now(tz='UTC').isoformat()}
    save_json(Path(cfg['output_dir'])/'environment.json',report)
    return report

def prepare_cohort(cfg, verify_hashes=False):
    m=pd.read_csv(cfg['manifest'])
    assert m.filename.is_unique, 'Duplicate filename mapping'
    assert m.tb_status.isin([0,1]).all(), 'Unexpected label'
    paths=[Path(cfg['data_root'])/str(p) for p in m.path]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError(f'{len(missing)} audio missing, e.g. {missing[:2]}')
    if verify_hashes:
        for p,expected in zip(paths,m.sha256):
            if sha256(p)!=expected:raise ValueError(f'Audio changed after audit: {p}')
    clinical=pd.read_csv(Path(cfg['data_root'])/'meta_data/Clinical/CODA_TB_Clinical_Meta_Info.csv')
    extra=pd.read_csv(Path(cfg['data_root'])/'meta_data/Clinical/CODA_TB_additional_variables_train.csv')
    assert clinical.participant.is_unique and extra.participant.is_unique
    # Read clinical source afresh rather than silently using an edited cached join.
    participants=clinical[clinical.participant.isin(m.participant)].merge(extra,on='participant',validate='one_to_one')
    participants=participants.sort_values('participant').reset_index(drop=True)
    assert participants.participant.nunique()==m.participant.nunique()
    labels=participants.set_index('participant').tb_status
    if not m.tb_status.eq(m.participant.map(labels)).all():raise ValueError('Manifest/source label mismatch')
    participants['hr_review']=~participants.heart_rate.between(30,220)
    participants['reference_review']=participants.Microbiologicreferencestandard.isna()
    if cfg['policy']['heart_rate_outside_30_220']=='set_missing':
        participants.loc[participants.hr_review,'heart_rate']=np.nan
    if cfg['policy']['missing_reference']=='exclude':
        participants=participants[~participants.reference_review].reset_index(drop=True)
    if cfg['policy']['clipping']=='exclude_above_1pct':
        m=m[m.clipping_fraction<=.01].copy()
        participants=participants[participants.participant.isin(m.participant)].reset_index(drop=True)
    m=m[m.participant.isin(participants.participant)].copy()
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    participants.to_csv(out/'cohort.csv',index=False);m.to_csv(out/'audio_manifest.csv',index=False)
    save_json(out/'data_provenance.json',{'manifest_sha256':sha256(cfg['manifest']),
        'source_metadata_sha256':{p.name:sha256(p) for p in (Path(cfg['data_root'])/'meta_data/Clinical').glob('*.csv')},
        'n_participants':len(participants),'n_coughs':len(m),'policy':cfg['policy'],
        'notice':'Third-party partial mirror. Distribution license unresolved. No clinical certification.'})
    return participants,m

def make_splits(participants,cfg):
    ids=participants.participant.to_numpy();y=participants.tb_status.to_numpy()
    development,test=train_test_split(ids,test_size=cfg['test_fraction'],stratify=y,random_state=cfg['seed'])
    labels=participants.set_index('participant').tb_status
    proper,cal=train_test_split(development,test_size=cfg['calibration_fraction']/(1-cfg['test_fraction']),
        stratify=labels.loc[development],random_state=cfg['seed']+1)
    split={'proper_train':sorted(proper.tolist()),'calibration':sorted(cal.tolist()),'test':sorted(test.tolist())}
    for k,a in split.items():
        for j,b in split.items():
            if k!=j:assert not set(a)&set(b)
    cv=StratifiedKFold(cfg['cv_folds'],shuffle=True,random_state=cfg['seed'])
    folds=[];proper=np.array(split['proper_train'])
    for tr,va in cv.split(proper,labels.loc[proper]):
        folds.append({'train':proper[tr].tolist(),'validation':proper[va].tolist()})
    split['cv_folds']=folds
    frozen=Path(cfg['package_root'])/'manifests/initial_participant_splits.json'
    if frozen.exists() and cfg.get('enforce_frozen_split',True):
        reference=json.loads(frozen.read_text())
        reference_ids=set(reference['proper_train'])|set(reference['calibration'])|set(reference['test'])
        if set(ids)==reference_ids and split!=reference:
            raise ValueError('Frozen participant split changed. Record a protocol amendment instead of tuning splits to test performance.')
    save_json(Path(cfg['output_dir'])/'splits.json',split)
    return split

def eda(participants,m,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(cfg['output_dir'])/'eda';out.mkdir(exist_ok=True,parents=True)
    counts=m.groupby('participant').size().rename('cough_count')
    table=participants.join(counts,on='participant')
    table.select_dtypes('number').describe().to_csv(out/'numeric_summary.csv')
    table.isna().sum().rename('missing').to_csv(out/'missing_values.csv')
    ct=pd.crosstab(table.Country,table.tb_status).reindex(columns=[0,1],fill_value=0)
    ct.to_csv(out/'country_label_counts.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    table.tb_status.value_counts().sort_index().plot.bar(ax=axes[0,0],color=['#287d8e','#d37537']);axes[0,0].set_title('Participants by TB label');axes[0,0].set_xlabel('0 = non-TB, 1 = TB')
    ct.plot.bar(stacked=True,ax=axes[0,1]);axes[0,1].set_title('Country and label (available subset)')
    table.heart_rate.hist(bins=25,ax=axes[1,0]);axes[1,0].set_title('HR after predefined review policy');axes[1,0].set_xlabel('BPM')
    table.cough_count.hist(bins=25,ax=axes[1,1]);axes[1,1].set_title('Coughs per participant')
    fig.tight_layout();fig.savefig(out/'cohort_overview.png',dpi=160);plt.close(fig)
    return table

def read_audio(path,cfg):
    with wave.open(str(path),'rb') as w:
        if w.getnchannels()!=1 or w.getsampwidth()!=2:raise ValueError('Expected mono PCM16 WAV')
        sr=w.getframerate();x=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32)/32768
    target=cfg['sample_rate'];g=math.gcd(sr,target)
    if sr!=target:x=signal.resample_poly(x,target//g,sr//g).astype(np.float32)
    n=round(target*cfg['window_seconds'])
    x=x[:n];x=np.pad(x,(0,max(0,n-len(x))))
    # Fixed per-record standardization; no statistics from validation/test cohorts.
    x=x-x.mean();x=x/max(float(x.std()),1e-5)
    return x.astype(np.float32)

def acoustic_features(x,sr):
    """Explicit SciPy MFCC/spectral baseline; not an exact copy of Sensors features."""
    nfft=512;win=round(.025*sr);hop=round(.010*sr)
    _,_,z=signal.stft(x,fs=sr,nperseg=win,noverlap=win-hop,nfft=nfft,boundary=None,padded=False)
    power=np.abs(z)**2+1e-12;freq=np.linspace(0,sr/2,power.shape[0])
    hz=lambda m:700*(10**(m/2595)-1)
    edges=hz(np.linspace(2595*np.log10(1+50/700),2595*np.log10(1+(sr/2)/700),42))
    bank=np.array([np.maximum(0,np.minimum((freq-edges[i])/(edges[i+1]-edges[i]),(edges[i+2]-freq)/(edges[i+2]-edges[i+1]))) for i in range(40)])
    mfcc=dct(np.log(bank@power+1e-12),type=2,axis=0,norm='ortho')[:13]
    delta=np.gradient(mfcc,axis=1);delta2=np.gradient(delta,axis=1)
    total=power.sum(axis=0);centroid=(power*freq[:,None]).sum(axis=0)/total
    bandwidth=np.sqrt((power*(freq[:,None]-centroid)**2).sum(axis=0)/total)
    flatness=np.exp(np.log(power).mean(axis=0))/power.mean(axis=0)
    entropy=-(power/total*np.log(power/total)).sum(axis=0)
    f=np.vstack([mfcc,delta,delta2,centroid[None],bandwidth[None],flatness[None],entropy[None]])
    names=[f'{group}{i}' for group in ['mfcc','delta','delta2'] for i in range(13)]+['centroid','bandwidth','flatness','entropy']
    values=np.concatenate([f.mean(axis=1),f.std(axis=1)])
    return values,[f'audio_{stat}_{name}' for stat in ['mean','std'] for name in names]

def prepare_audio_cache(m,cfg):
    cache=Path(cfg['package_root'])/'cache';cache.mkdir(exist_ok=True)
    fingerprint=hashlib.sha256((json.dumps({'sr':cfg['sample_rate'],'window':cfg['window_seconds'],'version':VERSION,'source_sha256':sha256(__file__)},sort_keys=True)+''.join(m.sort_values('filename').sha256)).encode()).hexdigest()
    folder=cache/fingerprint[:16];folder.mkdir(exist_ok=True)
    ordered=m.sort_values('filename').reset_index(drop=True)
    if (folder/'complete.json').exists():
        meta=json.loads((folder/'complete.json').read_text())
        if meta['fingerprint']==fingerprint and len(pd.read_csv(folder/'index.csv'))==len(m):
            return np.load(folder/'waveforms.npy',mmap_mode='r'),pd.read_csv(folder/'acoustic.csv'),ordered,folder
    n=round(cfg['sample_rate']*cfg['window_seconds'])
    waves=np.lib.format.open_memmap(folder/'waveforms.npy',mode='w+',dtype='float32',shape=(len(ordered),n))
    rows=[]
    for i,row in ordered.iterrows():
        x=read_audio(Path(cfg['data_root'])/row.path,cfg);waves[i]=x
        f,names=acoustic_features(x,cfg['sample_rate']);rows.append(f)
        if (i+1)%1000==0:print(f'Preprocessed {i+1}/{len(ordered)}',flush=True)
    waves.flush();features=pd.DataFrame(rows,columns=names)
    features.insert(0,'filename',ordered.filename.values);features.insert(0,'participant',ordered.participant.values)
    features.to_csv(folder/'acoustic.csv',index=False);ordered[['participant','filename','sha256']].to_csv(folder/'index.csv',index=False)
    save_json(folder/'complete.json',{'fingerprint':fingerprint,'shape':[len(ordered),n],'sample_rate':cfg['sample_rate'],'version':VERSION})
    return np.load(folder/'waveforms.npy',mmap_mode='r'),features,ordered,folder

def participant_features(participants,acoustic):
    cols=[c for c in acoustic if c.startswith('audio_')]
    # Equal participant weight; no cough-count or filename features.
    pooled=acoustic.groupby('participant')[cols].mean()
    return participants.set_index('participant').join(pooled,how='inner').sort_index()

def feature_columns(table,cfg,modality='fused',feature_set='mobile',without_hr=False):
    clinical=cfg['clinical_features'][feature_set]
    if without_hr:clinical=[c for c in clinical if c!='heart_rate']
    audio=[c for c in table if c.startswith('audio_')]
    cols=audio if modality=='audio' else clinical if modality=='clinical' else audio+clinical
    forbidden={'participant','tb_status','Country','HIVstatus','Microbiologicreferencestandard','Sputumxpertreferencestandard','Xpertcombinedsemiquant'}
    assert not set(cols)&forbidden
    return list(cols)

def make_preprocessor(cols):
    numeric=[c for c in cols if c in NUMERIC or c.startswith('audio_')];categorical=[c for c in cols if c not in numeric]
    transformations=[]
    if numeric:transformations.append(('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())]),numeric))
    if categorical:transformations.append(('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical))
    return ColumnTransformer(transformations,remainder='drop',sparse_threshold=0)

def metrics(y,p,threshold=.5):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'n':len(y),'roc_auc':roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
        'pr_auc':average_precision_score(y,p) if y.sum() else np.nan,'brier':brier_score_loss(y,p),
        'log_loss':log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1]),'threshold':threshold,
        'sensitivity':tp/(tp+fn) if tp+fn else np.nan,'specificity':tn/(tn+fp) if tn+fp else np.nan,
        'ppv':tp/(tp+fp) if tp+fp else np.nan,'npv':tn/(tn+fn) if tn+fn else np.nan,
        'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def bootstrap_metrics(y,p,threshold,repeats,seed):
    rng=np.random.default_rng(seed);y=np.asarray(y);p=np.asarray(p);rows=[]
    for _ in range(repeats):
        i=rng.integers(0,len(y),len(y))
        if len(np.unique(y[i]))==2:rows.append(metrics(y[i],p[i],threshold))
    df=pd.DataFrame(rows)
    return {k:{'low':float(df[k].quantile(.025)),'high':float(df[k].quantile(.975))} for k in ['roc_auc','pr_auc','sensitivity','specificity','brier']}

def classical_candidates(cfg,include_reference=True):
    candidates=[]
    specs=[('mobile',m,False) for m in ['audio','clinical','fused']]+[('mobile','fused',True)]
    if include_reference and cfg['run_reference_comparator']:specs.append(('reference','fused',False))
    for fs,mod,nohr in specs:
        for c in cfg['lr_C']:
            candidates.append({'family':'lr','feature_set':fs,'modality':mod,'without_hr':nohr,'params':{'C':c}})
        for params in cfg['catboost_trials']:
            candidates.append({'family':'catboost','feature_set':fs,'modality':mod,'without_hr':nohr,'params':params})
    return candidates

def fit_classical(table,train_ids,candidate,cfg):
    from catboost import CatBoostClassifier
    cols=feature_columns(table,cfg,candidate['modality'],candidate['feature_set'],candidate['without_hr'])
    pp=make_preprocessor(cols);x=pp.fit_transform(table.loc[train_ids,cols]);y=table.loc[train_ids,'tb_status'].astype(int)
    if candidate['family']=='lr':model=LogisticRegression(**candidate['params'],max_iter=3000,class_weight='balanced',random_state=cfg['seed'])
    else:model=CatBoostClassifier(**candidate['params'],loss_function='Logloss',auto_class_weights='Balanced',random_seed=cfg['seed'],thread_count=4,verbose=False,allow_writing_files=False)
    model.fit(x,y)
    return {'family':candidate['family'],'model':model,'preprocessor':pp,'columns':cols,'candidate':candidate}

def predict_classical(bundle,table,ids):
    x=bundle['preprocessor'].transform(table.loc[ids,bundle['columns']])
    return bundle['model'].predict_proba(x)[:,1]

def search_classical(table,splits,cfg):
    out=Path(cfg['output_dir']);out.mkdir(exist_ok=True,parents=True);scores=[];predictions=[]
    for number,c in enumerate(classical_candidates(cfg)):
        print('Classical candidate',number+1,c,flush=True)
        vals=[]
        for fold,sp in enumerate(splits['cv_folds']):
            bundle=fit_classical(table,sp['train'],c,cfg);p=predict_classical(bundle,table,sp['validation'])
            y=table.loc[sp['validation'],'tb_status'];vals.append(metrics(y,p)['roc_auc'])
            predictions.extend({'candidate':number,'fold':fold,'participant':pid,'y':int(yy),'raw_probability':float(pp)} for pid,yy,pp in zip(sp['validation'],y,p))
        scores.append({'candidate':number,'spec':c,'cv_auc_mean':float(np.mean(vals)),'cv_auc_std':float(np.std(vals))})
        save_json(out/'classical_search.json',scores);pd.DataFrame(predictions).to_csv(out/'classical_oof.csv',index=False)
    return scores

# Independent adaptation: five CNN blocks, BiGRU, temporal attention, clinical fusion.
# Pooling strides, BCE loss and mean patient aggregation are explicit adaptations.
import torch
from torch import nn

class CoughFusionNet(nn.Module):
    def __init__(self,clinical_dim,hidden=24,dropout=.2,modality='fused'):
        super().__init__();blocks=[];previous=1
        for kernel,stride in zip([24,24,24,24,48],[4,4,4,4,2]):
            blocks.extend([nn.Conv1d(previous,12,3,padding=1),nn.LeakyReLU(.01),
                nn.Conv1d(12,12,3,padding=1),nn.LeakyReLU(.01),
                nn.Conv1d(12,12,kernel,stride=stride,padding=kernel//2),nn.LeakyReLU(.01),nn.Dropout(dropout)])
            previous=12
        self.encoder=nn.Sequential(*blocks)
        self.gru=nn.GRU(12,hidden,batch_first=True,bidirectional=True)
        self.attention=nn.Sequential(nn.Linear(hidden*2,32),nn.Tanh(),nn.Linear(32,1))
        self.modality=modality
        self.head=nn.Sequential(nn.Linear(hidden*2+(clinical_dim if modality=='fused' else 0),64),
            nn.LeakyReLU(.01),nn.Dropout(dropout),nn.Linear(64,1))
    def forward(self,waveforms,clinical):
        h=self.encoder(waveforms.unsqueeze(1)).transpose(1,2);h,_=self.gru(h)
        weight=torch.softmax(self.attention(h),dim=1);pooled=(h*weight).sum(dim=1)
        if self.modality=='fused':pooled=torch.cat([pooled,clinical],dim=1)
        return self.head(pooled).squeeze(-1)

def deep_predict(bundle,table,ids,waves,index,cfg):
    model=bundle['model'];device=next(model.parameters()).device;model.eval()
    wanted=set(ids);positions=np.flatnonzero(index.participant.isin(wanted).to_numpy())
    clinical=bundle['preprocessor'].transform(table.loc[index.iloc[positions].participant,bundle['columns']]).astype('float32')
    result=[]
    with torch.no_grad():
        for begin in range(0,len(positions),cfg['deep_batch_size']):
            pos=positions[begin:begin+cfg['deep_batch_size']]
            xx=torch.as_tensor(np.array(waves[pos]),device=device)
            cc=torch.as_tensor(clinical[begin:begin+len(pos)],device=device)
            result.extend(torch.sigmoid(model(xx,cc)).cpu().numpy().tolist())
    pred=pd.DataFrame({'participant':index.iloc[positions].participant.values,'p':result}).groupby('participant').p.mean()
    return pred.loc[ids].to_numpy()

def fit_deep(table,train_ids,candidate,waves,index,cfg,epochs=None):
    seed_everything(cfg['seed']);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ids=list(train_ids);labels=table.tb_status
    if epochs is None:
        fit_ids,early_ids=train_test_split(train_ids,test_size=.2,stratify=labels.loc[train_ids],random_state=cfg['seed']+21)
    else:fit_ids,early_ids=train_ids,[]
    cols=cfg['clinical_features'][candidate['feature_set']]
    if candidate.get('without_hr'):cols=[x for x in cols if x!='heart_rate']
    pp=make_preprocessor(cols);pp.fit(table.loc[fit_ids,cols])
    params=candidate['params'];dim=pp.transform(table.loc[fit_ids[:1],cols]).shape[1]
    model=CoughFusionNet(dim,hidden=params['hidden'],dropout=params['dropout'],modality=candidate['modality']).to(device)
    bundle={'family':'deep','model':model,'preprocessor':pp,'columns':cols,'candidate':candidate,'clinical_dim':dim}
    optimizer=torch.optim.AdamW(model.parameters(),lr=params['lr'],weight_decay=params['weight_decay'])
    positions={pid:np.flatnonzero(index.participant.eq(pid).to_numpy()) for pid in fit_ids}
    if any(len(x)==0 for x in positions.values()):raise ValueError('Participant has no cached cough')
    all_clinical=pp.transform(table.loc[index.participant,cols]).astype('float32')
    positive=float(labels.loc[fit_ids].sum());ratio=(len(fit_ids)-positive)/max(positive,1)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio,device=device),reduction='none')
    best=-np.inf;best_state=None;best_epoch=0;stale=0;history=[];rng=np.random.default_rng(cfg['seed'])
    for epoch in range(epochs or cfg['deep_max_epochs']):
        model.train();selected=[];weights=[]
        for pid,pos in positions.items():
            chosen=rng.choice(pos,min(len(pos),cfg['deep_train_cough_cap']),replace=False)
            selected.extend(chosen);weights.extend([1/len(chosen)]*len(chosen))
        order=rng.permutation(len(selected));selected=np.asarray(selected)[order];weights=np.asarray(weights,dtype='float32')[order]
        total_loss=0.;total_weight=0.
        for start in range(0,len(selected),cfg['deep_batch_size']):
            pos=selected[start:start+cfg['deep_batch_size']];w=torch.as_tensor(weights[start:start+len(pos)],device=device)
            xx=torch.as_tensor(np.array(waves[pos]),device=device);cc=torch.as_tensor(all_clinical[pos],device=device)
            yy=torch.tensor(labels.loc[index.iloc[pos].participant].to_numpy(dtype='float32'),device=device)
            optimizer.zero_grad();losses=loss_fn(model(xx,cc),yy)
            loss=(losses*w).sum()/w.sum();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total_loss+=float((losses.detach()*w).sum());total_weight+=float(w.sum())
        auc=None
        if early_ids:
            prob=deep_predict(bundle,table,list(early_ids),waves,index,cfg);auc=roc_auc_score(labels.loc[early_ids],prob)
            if auc>best+1e-5:
                best=auc;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
            else:stale+=1
        history.append({'epoch':epoch+1,'loss':total_loss/max(total_weight,1),'early_stop_auc':auc})
        if early_ids and stale>=cfg['deep_patience']:break
    if best_state is not None:model.load_state_dict(best_state)
    bundle['best_epoch']=best_epoch if early_ids else epochs;bundle['history']=history
    return bundle

def search_deep(table,splits,waves,index,cfg):
    if not cfg['run_deep']:return []
    if cfg['require_cuda_for_full_training'] and not torch.cuda.is_available():raise RuntimeError('Full deep training requires CUDA. Run preflight here and full notebook on RTX PC.')
    scores=[];oof=[];out=Path(cfg['output_dir'])
    candidates=[{'family':'deep','feature_set':'mobile','modality':m,'without_hr':False,'params':p} for m in ['audio','fused'] for p in cfg['deep_trials']]
    for number,c in enumerate(candidates):
        vals=[];epochs=[]
        print('Deep candidate',number+1,c,flush=True)
        for fold,sp in enumerate(splits['cv_folds']):
            bundle=fit_deep(table,sp['train'],c,waves,index,cfg)
            pred=deep_predict(bundle,table,sp['validation'],waves,index,cfg)
            vals.append(roc_auc_score(table.loc[sp['validation'],'tb_status'],pred));epochs.append(bundle['best_epoch'])
            oof.extend({'candidate':number,'fold':fold,'participant':pid,'y':int(table.loc[pid,'tb_status']),'raw_probability':float(prob)} for pid,prob in zip(sp['validation'],pred))
            save_json(out/f'deep_candidate_{number}_fold_{fold}_history.json',bundle['history'])
        scores.append({'candidate':number,'spec':c,'cv_auc_mean':float(np.mean(vals)),'cv_auc_std':float(np.std(vals)),
            'final_epochs':max(1,int(np.median(epochs))),'fold_best_epochs':epochs})
        save_json(out/'deep_search.json',scores);pd.DataFrame(oof).to_csv(out/'deep_oof.csv',index=False)
    return scores

def fit_calibrator(y,p,cfg):
    # Small independent calibration cohort: regularized Platt calibration, not isotonic.
    logit=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1))[:,None]
    cal=LogisticRegression(C=1.,max_iter=2000,random_state=cfg['seed']);cal.fit(logit,y)
    pp=cal.predict_proba(logit)[:,1]
    # Choose operational threshold only on calibration IDs. Test sensitivity can differ.
    fpr,tpr,threshold=roc_curve(y,pp);good=np.flatnonzero(tpr>=cfg['target_sensitivity'])
    chosen=good[np.argmin(fpr[good])]
    return cal,float(threshold[chosen])

def calibrated_probability(cal,p):
    p=np.asarray(p);logit=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1))[:,None]
    return cal.predict_proba(logit)[:,1]

def save_bundle(bundle,directory,cfg):
    directory=Path(directory);directory.mkdir(exist_ok=True,parents=True)
    if bundle['family']=='deep':
        torch.save({k:v.detach().cpu() for k,v in bundle['model'].state_dict().items()},directory/'weights.pt')
        joblib.dump(bundle['preprocessor'],directory/'preprocessor.joblib')
        meta={k:bundle[k] for k in ['family','columns','candidate','clinical_dim','best_epoch']}
    else:
        joblib.dump(bundle,directory/'model.joblib');meta={k:bundle[k] for k in ['family','columns','candidate']}
    joblib.dump(bundle['calibrator'],directory/'calibrator.joblib')
    meta.update(threshold=bundle['threshold'],sample_rate=cfg['sample_rate'],window_seconds=cfg['window_seconds'],
        aggregation='mean_probability_per_participant',implementation='independent adaptation, not official winner',version=VERSION)
    save_json(directory/'metadata.json',meta)

def load_bundle(directory,device='cpu'):
    directory=Path(directory);meta=json.loads((directory/'metadata.json').read_text())
    if meta['family']=='deep':
        c=meta['candidate'];model=CoughFusionNet(meta['clinical_dim'],c['params']['hidden'],c['params']['dropout'],c['modality'])
        model.load_state_dict(torch.load(directory/'weights.pt',map_location=device,weights_only=True));model.to(device);model.eval()
        bundle={**meta,'model':model,'preprocessor':joblib.load(directory/'preprocessor.joblib')}
    else:bundle=joblib.load(directory/'model.joblib');bundle['threshold']=meta['threshold']
    bundle['calibrator']=joblib.load(directory/'calibrator.joblib')
    return bundle

def predict_bundle(bundle,table,ids,waves,index,cfg,calibrated=True):
    raw=deep_predict(bundle,table,ids,waves,index,cfg) if bundle['family']=='deep' else predict_classical(bundle,table,ids)
    return calibrated_probability(bundle['calibrator'],raw) if calibrated else raw

def finalize_models(table,splits,classical_scores,deep_scores,waves,index,cfg):
    scores=classical_scores+deep_scores
    def key(s):
        c=s['spec'];return (c['family'],c['feature_set'],c['modality'],c.get('without_hr',False))
    groups={}
    for s in scores:
        if key(s) not in groups or s['cv_auc_mean']>groups[key(s)]['cv_auc_mean']:groups[key(s)]=s
    primary_candidates=[s for s in groups.values() if s['spec']['feature_set']=='mobile' and s['spec']['modality']=='fused' and not s['spec'].get('without_hr')]
    selected=max(primary_candidates,key=lambda s:s['cv_auc_mean'])
    fallback=max([s for s in groups.values() if s['spec'].get('without_hr')],key=lambda s:s['cv_auc_mean'])
    out=Path(cfg['output_dir']);summary=[];bundles={};selected_name=fallback_name=None
    for key_,s in groups.items():
        c=s['spec'];name='_'.join(map(str,key_));print('Final fitting',name,flush=True)
        if c['family']=='deep':bundle=fit_deep(table,splits['proper_train'],c,waves,index,cfg,epochs=s['final_epochs'])
        else:bundle=fit_classical(table,splits['proper_train'],c,cfg)
        raw=lambda ids: predict_bundle(bundle,table,ids,waves,index,cfg,calibrated=False)
        pcal=raw(splits['calibration']);cal,threshold=fit_calibrator(table.loc[splits['calibration'],'tb_status'],pcal,cfg)
        bundle.update(calibrator=cal,threshold=threshold)
        folder=out/'models'/name;save_bundle(bundle,folder,cfg)
        for part in ['calibration','test']:
            ids=splits[part];praw=pcal if part=='calibration' else raw(ids);prob=calibrated_probability(cal,praw)
            pd.DataFrame({'participant':ids,'y':table.loc[ids,'tb_status'].values,'Country':table.loc[ids,'Country'].values,
                'raw_probability':praw,'probability':prob,'prediction':(prob>=threshold).astype(int)}).to_csv(folder/f'{part}_predictions.csv',index=False)
            if part=='test':
                result=metrics(table.loc[ids,'tb_status'],prob,threshold)
                result['bootstrap_ci']=bootstrap_metrics(table.loc[ids,'tb_status'],prob,threshold,cfg['bootstrap_repeats'],cfg['seed'])
                save_json(folder/'test_metrics.json',result)
                summary.append({'model':name,'cv_auc_mean':s['cv_auc_mean'],'cv_auc_std':s['cv_auc_std'],**{k:v for k,v in result.items() if k!='bootstrap_ci'}})
        bundles[name]=bundle
        if s is selected:selected_name=name
        if s is fallback:fallback_name=name
    pd.DataFrame(summary).to_csv(out/'model_comparison_test.csv',index=False)
    selection={'primary':selected_name,'fallback_without_hr':fallback_name,'selection_criterion':'Highest mean development CV ROC-AUC among MOBILE FUSED candidates; test never used for selection',
        'primary_cv_auc':selected['cv_auc_mean'],'test_used_for_selection':False,'no_clinical_validation':True}
    save_json(out/'selection.json',selection)
    return bundles,selection

def plot_results(cfg,selection):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    from sklearn.calibration import calibration_curve
    out=Path(cfg['output_dir']);name=selection['primary'];pred=pd.read_csv(out/'models'/name/'test_predictions.csv')
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    RocCurveDisplay.from_predictions(pred.y,pred.probability,ax=ax[0]);PrecisionRecallDisplay.from_predictions(pred.y,pred.probability,ax=ax[1])
    actual,estimated=calibration_curve(pred.y,pred.probability,n_bins=5,strategy='quantile')
    ax[2].plot(estimated,actual,'o-');ax[2].plot([0,1],[0,1],'--');ax[2].set(xlabel='Predicted risk',ylabel='Observed fraction',title='Test calibration (small bins)')
    fig.tight_layout();fig.savefig(out/'selected_model_test_curves.png',dpi=160);plt.close(fig)
    rows=[];threshold=json.loads((out/'models'/name/'metadata.json').read_text())['threshold']
    for country,g in pred.groupby('Country'):rows.append({'Country':country,**metrics(g.y,g.probability,threshold)})
    pd.DataFrame(rows).to_csv(out/'test_country_subgroups.csv',index=False)

def run_country_validation(table,cfg,classical_scores):
    """Predeclared domain experiment. Nested selection inside each 6-country training set."""
    if not cfg['run_country_validation']:return
    # Fixed family/hyperparameter grid, never choose this grid from held-country labels.
    candidates=[c for c in classical_candidates(cfg,include_reference=False) if c['modality']=='fused' and not c['without_hr']]
    rows=[];out=Path(cfg['output_dir'])/'country_validation';out.mkdir(exist_ok=True)
    for country in sorted(table.Country.unique()):
        outer_train=table.index[table.Country!=country].to_numpy();outer_test=table.index[table.Country==country].tolist()
        proper,cal=train_test_split(outer_train,test_size=.15,stratify=table.loc[outer_train,'tb_status'],random_state=cfg['seed'])
        cv=StratifiedKFold(3,shuffle=True,random_state=cfg['seed']);best_score=-np.inf;best_spec=None
        for c in candidates:
            vals=[]
            for tr,va in cv.split(proper,table.loc[proper,'tb_status']):
                b=fit_classical(table,proper[tr].tolist(),c,cfg);p=predict_classical(b,table,proper[va].tolist())
                vals.append(roc_auc_score(table.loc[proper[va],'tb_status'],p))
            if np.mean(vals)>best_score:best_score=float(np.mean(vals));best_spec=c
        b=fit_classical(table,proper.tolist(),best_spec,cfg);pcal=predict_classical(b,table,cal.tolist())
        calibrator,threshold=fit_calibrator(table.loc[cal,'tb_status'],pcal,cfg)
        p=calibrated_probability(calibrator,predict_classical(b,table,outer_test))
        rows.append({'held_country':country,'selection_cv_auc':best_score,'candidate':json.dumps(best_spec),**metrics(table.loc[outer_test,'tb_status'],p,threshold)})
        pd.DataFrame({'participant':outer_test,'y':table.loc[outer_test,'tb_status'].values,'probability':p}).to_csv(out/f'{country}_predictions.csv',index=False)
        pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
        print('Held country complete',country,flush=True)

def policy_sensitivity(participants,acoustic,splits,cfg,classical_scores):
    if not cfg['run_policy_sensitivity']:return
    source=pd.read_csv(Path(cfg['data_root'])/'meta_data/Clinical/CODA_TB_Clinical_Meta_Info.csv').set_index('participant')
    primary=max([s for s in classical_scores if s['spec']['feature_set']=='mobile' and s['spec']['modality']=='fused' and not s['spec'].get('without_hr')],key=lambda s:s['cv_auc_mean'])
    rows=[]
    variants=['primary_policy','retain_source_hr','exclude_missing_reference','exclude_clipped_coughs']
    for variant in variants:
        pp=participants.copy();aa=acoustic.copy()
        if variant=='retain_source_hr':pp['heart_rate']=pp.participant.map(source.heart_rate)
        if variant=='exclude_missing_reference':pp=pp[~pp.reference_review]
        if variant=='exclude_clipped_coughs':
            m=pd.read_csv(Path(cfg['output_dir'])/'audio_manifest.csv');aa=aa[aa.filename.isin(m.loc[m.clipping_fraction<=.01,'filename'])]
            pp=pp[pp.participant.isin(aa.participant)]
        table=participant_features(pp,aa);prob=[];labels=[];ids_out=[]
        for sp in splits['cv_folds']:
            train=[x for x in sp['train'] if x in table.index];val=[x for x in sp['validation'] if x in table.index]
            b=fit_classical(table,train,primary['spec'],cfg);p=predict_classical(b,table,val)
            prob.extend(p);labels.extend(table.loc[val,'tb_status']);ids_out.extend(val)
        rows.append({'policy':variant,'n_participants_development':len(ids_out),**metrics(labels,prob)})
    pd.DataFrame(rows).to_csv(Path(cfg['output_dir'])/'policy_sensitivity_development_only.csv',index=False)

def hr_stress_test(bundle,table,splits,waves,index,cfg):
    ids=splits['test'];rows=[]
    for shift in [0,*cfg['xai_hr_shifts_bpm'],'missing']:
        changed=table.copy()
        changed.loc[ids,'heart_rate']=np.nan if shift=='missing' else changed.loc[ids,'heart_rate']+shift
        pred=predict_bundle(bundle,changed,ids,waves,index,cfg)
        rows.append({'hr_condition':str(shift),**metrics(table.loc[ids,'tb_status'],pred,bundle['threshold'])})
    pd.DataFrame(rows).to_csv(Path(cfg['output_dir'])/'hr_stress_test.csv',index=False)

def _rank_correlation(a,b):
    from scipy.stats import spearmanr
    if np.std(a)<1e-12 or np.std(b)<1e-12:return np.nan
    return float(spearmanr(a,b).statistic)

def classical_xai(bundle,table,ids,background_ids,cfg,folder):
    from catboost import Pool
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    pp=bundle['preprocessor'];cols=bundle['columns'];names=pp.get_feature_names_out().tolist()
    x=pp.transform(table.loc[ids,cols]);bg=pp.transform(table.loc[background_ids,cols]);reference=bg.mean(axis=0)
    rng=np.random.default_rng(cfg['seed']);bg_positions=rng.choice(len(bg),min(128,len(bg)),replace=False)
    bg_y=table.loc[list(background_ids),'tb_status'].to_numpy()[bg_positions];bg=bg[bg_positions]
    def values(xx):
        if bundle['family']=='catboost':
            return bundle['model'].get_feature_importance(Pool(xx,label=table.loc[ids,'tb_status'].to_numpy()),
                type='ShapValues',reference_data=Pool(bg,label=bg_y))
        val=(xx-reference)*bundle['model'].coef_[0]
        base=float(bundle['model'].intercept_[0]+reference@bundle['model'].coef_[0])
        return np.c_[val,np.full(len(xx),base)]
    tic=time.perf_counter();shap=values(x);elapsed=time.perf_counter()-tic
    # Platt scaling is affine in raw log-odds: this transformation preserves additivity.
    scale=float(bundle['calibrator'].coef_[0,0]);intercept=float(bundle['calibrator'].intercept_[0])
    contributions=shap[:,:-1]*scale;base=shap[:,-1]*scale+intercept
    raw=bundle['model'].predict_proba(x)[:,1];prob=calibrated_probability(bundle['calibrator'],raw)
    reconstructed=1/(1+np.exp(-np.clip(base+contributions.sum(axis=1),-50,50)))
    pd.DataFrame(contributions,index=ids,columns=names).rename_axis('participant').to_csv(folder/'calibrated_log_odds_attributions.csv')
    save_json(folder/'explanation_definition.json',{'method':'CatBoost native SHAP with training background' if bundle['family']=='catboost' else 'Linear interventional attribution with training mean',
        'output_space':'calibrated log odds, NOT probability percentages','background_ids':list(background_ids),
        'max_probability_reconstruction_error':float(np.max(np.abs(prob-reconstructed))),
        'calibration_clipping_warning':'Native log-odds transformation can cease to be exact if probability clipping is active. Use grouped_probability_attributions.csv for attribution of the final calibrated pipeline.',
        'total_explanation_seconds':elapsed,'n':len(ids),'warning':'Attribution is model behavior, not a causal medical explanation. Feature masking can create off-manifold inputs.'})
    audio=[i for i,n in enumerate(names) if 'audio_' in n];clinical=[i for i in range(len(names)) if i not in audio]
    pd.DataFrame({'participant':ids,'audio_signed_log_odds':contributions[:,audio].sum(axis=1),
        'clinical_signed_log_odds':contributions[:,clinical].sum(axis=1),'base_log_odds':base,'probability':prob}).to_csv(folder/'modality_contributions.csv',index=False)
    rows=[];k=max(1,len(names)//10)
    for i,pid in enumerate(ids):
        ranked=np.argsort(np.abs(contributions[i]))[-k:];random_idx=rng.choice(len(names),k,replace=False)
        for label,positions in [('important',ranked),('random',random_idx)]:
            altered=x[i:i+1].copy();altered[:,positions]=reference[positions]
            pert=calibrated_probability(bundle['calibrator'],bundle['model'].predict_proba(altered)[:,1])[0]
            rows.append({'participant':pid,'mask':label,'absolute_probability_change':abs(float(prob[i]-pert))})
    pd.DataFrame(rows).to_csv(folder/'faithfulness_encoded_feature_masking.csv',index=False)
    noisy=x.copy();numeric=[i for i,n in enumerate(names) if n.startswith('numeric__')]
    noisy[:,numeric]+=rng.normal(0,.01,(len(x),len(numeric)))
    other=values(noisy)[:,:-1]*scale
    pd.DataFrame({'participant':ids,'absolute_attribution_rank_spearman':[_rank_correlation(np.abs(a),np.abs(b)) for a,b in zip(contributions,other)]}).to_csv(folder/'stability_numeric_encoded_noise.csv',index=False)
    # Explain the COMPLETE calibrated pipeline in probability space, including
    # imputation/encoding and any clipping, via grouped Monte Carlo Shapley paths.
    groups={}
    audio_cols=[c for c in cols if c.startswith('audio_')]
    if audio_cols:groups['audio_all_acoustic_features']=audio_cols
    groups.update({c:[c] for c in cols if not c.startswith('audio_')})
    group_names=list(groups);n_paths=cfg.get('xai_permutations',32);group_rows=[]
    for pid in ids:
        background_choices=rng.choice(list(background_ids),n_paths,replace=True)
        path_rows=[];orders=[]
        for background_id in background_choices:
            row=table.loc[background_id,cols].copy();path_rows.append(row.copy())
            order=rng.permutation(len(group_names));orders.append(order)
            for g in order:
                group=groups[group_names[g]];row.loc[group]=table.loc[pid,group].values
                path_rows.append(row.copy())
        raw_rows=pd.DataFrame(path_rows,columns=cols)
        final_probs=calibrated_probability(bundle['calibrator'],bundle['model'].predict_proba(pp.transform(raw_rows))[:,1])
        paths=final_probs.reshape(n_paths,len(group_names)+1);v=np.zeros((n_paths,len(group_names)))
        for row_number,order in enumerate(orders):v[row_number,order]=np.diff(paths[row_number])
        averaged=v.mean(axis=0);baseline=float(paths[:,0].mean());target=float(paths[:,-1].mean())
        group_rows.append({'participant':pid,'base_probability':baseline,'probability':target,
            'completeness_error':target-baseline-float(averaged.sum()),**dict(zip(group_names,averaged)),
            **dict(zip(['mcse_'+g for g in group_names],v.std(axis=0,ddof=1)/math.sqrt(n_paths)))})
    pd.DataFrame(group_rows).to_csv(folder/'grouped_probability_attributions.csv',index=False)
    save_json(folder/'grouped_probability_definition.json',{'method':'Monte Carlo grouped interventional Shapley paths through the COMPLETE calibrated pipeline',
        'groups':groups,'permutations_per_participant':n_paths,'background':'Proper-training participant rows; sampled with replacement',
        'output_space':'Probability differences from the sampled training-background baseline; not medical causes',
        'caution':'Monte Carlo estimates have sampling error; mixed feature combinations can be off manifold. Audio is one joint group.'})

def _patient_output(model,w,c,cal):
    p=torch.sigmoid(model(w,c)).mean().clamp(1e-6,1-1e-6)
    z=torch.log(p/(1-p))*float(cal.coef_[0,0])+float(cal.intercept_[0])
    return torch.sigmoid(z)

def integrated_gradients_patient(model,w,c,bw,bc,cal,steps):
    model.eval();sw=torch.zeros_like(w);sc=torch.zeros_like(c)
    for i,alpha in enumerate(torch.linspace(0,1,steps+1,device=w.device)):
        wi=(bw+alpha*(w-bw)).detach().requires_grad_(True);ci=(bc+alpha*(c-bc)).detach().requires_grad_(True)
        # cuDNN RNN eval-mode backward is unsupported on CUDA. Use native GRU
        # for attribution while keeping dropout disabled and parameters fixed.
        with torch.backends.cudnn.flags(enabled=False):
            result=_patient_output(model,wi,ci,cal)
            gw,gc=torch.autograd.grad(result,(wi,ci),allow_unused=True)
        weight=.5 if i in [0,steps] else 1.
        if gw is not None:sw+=gw*weight
        if gc is not None:sc+=gc*weight
    aw=(w-bw)*sw/steps;ac=(c-bc)*sc/steps
    with torch.no_grad():delta=float(_patient_output(model,w,c,cal)-_patient_output(model,bw,bc,cal)-aw.sum()-ac.sum())
    return aw.detach(),ac.detach(),delta

def deep_xai(bundle,table,ids,background_ids,waves,index,cfg,folder):
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True);model=bundle['model'];model.eval();device=next(model.parameters()).device
    pp=bundle['preprocessor'];cols=bundle['columns'];names=pp.get_feature_names_out().tolist()
    reference=pp.transform(table.loc[background_ids,cols]).mean(axis=0).astype('float32')
    rng=np.random.default_rng(cfg['seed']);rows=[]
    for pid in ids:
        pos=np.flatnonzero(index.participant.eq(pid).to_numpy());w=torch.tensor(np.array(waves[pos]),device=device)
        cc=pp.transform(table.loc[[pid]*len(pos),cols]).astype('float32');c=torch.tensor(cc,device=device)
        bw=torch.zeros_like(w);bc=torch.tensor(np.repeat(reference[None],len(pos),axis=0),device=device)
        tic=time.perf_counter();steps=cfg['ig_steps']
        aw,ac,delta=integrated_gradients_patient(model,w,c,bw,bc,bundle['calibrator'],steps)
        while abs(delta)>.02 and steps<cfg['ig_steps_max']:
            steps=min(steps*2,cfg['ig_steps_max']);aw,ac,delta=integrated_gradients_patient(model,w,c,bw,bc,bundle['calibrator'],steps)
        elapsed=time.perf_counter()-tic
        original=float(_patient_output(model,w,c,bundle['calibrator']).detach())
        # Occlude contiguous 20-ms blocks, not isolated samples; compare with random.
        block=max(1,round(.02*cfg['sample_rate']));flat=aw.abs().cpu().numpy();segments=[]
        for cough in range(len(w)):
            for start in range(0,w.shape[1],block):segments.append((float(flat[cough,start:start+block].sum()),cough,start))
        k=max(1,len(segments)//10);ranked=sorted(segments,reverse=True)[:k]
        random_segs=[segments[i] for i in rng.choice(len(segments),k,replace=False)]
        changes={}
        with torch.no_grad():
            for label,chosen in [('important',ranked),('random',random_segs)]:
                altered=w.clone()
                for _,cough,start in chosen:altered[cough,start:start+block]=0
                changes[label]=abs(original-float(_patient_output(model,altered,c,bundle['calibrator'])))
        noise=torch.tensor(rng.normal(0,cfg['xai_stability_noise_std'],w.shape),dtype=w.dtype,device=device)
        noisy_w=w+noise
        aw2,_,_=integrated_gradients_patient(model,noisy_w,c,bw,bc,bundle['calibrator'],cfg['ig_steps'])
        stability=_rank_correlation(aw.abs().cpu().numpy().ravel(),aw2.abs().cpu().numpy().ravel())
        # Head reinitialization sanity check; not a clinical validity test.
        randomized=copy.deepcopy(model)
        for layer in randomized.head:
            if hasattr(layer,'reset_parameters'):layer.reset_parameters()
        ar,_,_=integrated_gradients_patient(randomized,w,c,bw,bc,bundle['calibrator'],cfg['ig_steps'])
        sanity=_rank_correlation(aw.abs().cpu().numpy().ravel(),ar.abs().cpu().numpy().ravel())
        np.savez_compressed(folder/f'{pid}_attribution.npz',audio_ig=aw.cpu().numpy(),clinical_ig=ac.cpu().numpy(),
            filenames=index.iloc[pos].filename.to_numpy(dtype=str),clinical_feature_names=np.array(names),waveforms=w.cpu().numpy())
        rows.append({'participant':pid,'probability':original,'audio_signed_probability_contribution':float(aw.sum()),
            'clinical_signed_probability_contribution':float(ac.sum()),'completeness_delta':delta,'ig_steps':steps,
            'explanation_seconds':elapsed,'important_mask_change':changes['important'],'random_mask_change':changes['random'],
            'noise_stability_spearman':stability,'randomized_head_spearman':sanity})
        pd.DataFrame(rows).to_csv(folder/'patient_explanations.csv',index=False)
    save_json(folder/'explanation_definition.json',{'method':'Integrated Gradients of calibrated mean-probability patient prediction',
        'audio_baseline':'zeros after waveform standardization','clinical_baseline':'mean encoded features of proper training participants',
        'background_ids':list(background_ids),'output_space':'probability differences relative to baseline, not causal factors',
        'warnings':['Baseline may be off manifold; disclose sensitivity.','Completeness residual is reported; >0.02 requires inspection.','Sanity check randomizes the output head only.','No clinical interpretation or validated audio biomarker is implied.']})

def run_xai(bundles,selection,table,splits,waves,index,cfg):
    # Deterministic label-independent selection, fixed before inspecting explanations.
    rng=np.random.default_rng(cfg['seed']+100)
    ids=sorted(rng.choice(splits['test'],min(cfg['xai_max_participants'],len(splits['test'])),replace=False).tolist())
    out=Path(cfg['output_dir'])/'xai';save_json(out/'explanation_ids.json',ids)
    selected=selection['primary'];bundle=bundles[selected]
    if bundle['family']=='deep':deep_xai(bundle,table,ids,splits['proper_train'],waves,index,cfg,out/selected)
    else:classical_xai(bundle,table,ids,splits['proper_train'],cfg,out/selected)
    # Always retain an interpretable tree/linear comparator even if deep is selected.
    for name,b in bundles.items():
        if name!=selected and b['family']=='catboost' and b['candidate']['feature_set']=='mobile' and b['candidate']['modality']=='fused' and not b['candidate'].get('without_hr'):
            classical_xai(b,table,ids,splits['proper_train'],cfg,out/name)
    hr_stress_test(bundle,table,splits,waves,index,cfg)

def parity_check(bundles,selection,table,splits,waves,index,cfg):
    out=Path(cfg['output_dir']);ids=splits['test'][:5];rows=[]
    for name in [selection['primary'],selection['fallback_without_hr']]:
        bundle=bundles[name];before=predict_bundle(bundle,table,ids,waves,index,cfg)
        loaded=load_bundle(out/'models'/name,'cpu');after=predict_bundle(loaded,table,ids,waves,index,cfg)
        error=float(np.max(np.abs(before-after)))
        if error>1e-4:raise AssertionError(f'Saved inference mismatch {name}: {error}')
        rows.append({'model':name,'max_abs_cpu_reload_error':error})
    save_json(out/'inference_parity.json',rows)

def export_return_bundle(cfg):
    out=Path(cfg['output_dir']);package=Path(cfg['package_root'])
    save_json(out/'RUN_COMPLETE.json',{'status':'completed','utc':pd.Timestamp.now(tz='UTC').isoformat(),
        'notice':'Research outputs only. Test remains a single cohort, not external clinical validation.',
        'source_sha256':sha256(__file__)})
    contents={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_checksums.json'}
    save_json(out/'artifact_checksums.json',contents)
    dest=out.parent/(out.name+'_RETURN.zip')
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,'run/'+p.relative_to(out).as_posix())
        for p in (package/'src').glob('*.py'):z.write(p,'research_package/src/'+p.name)
        for name in ['config.json','CODA_TB_Research.ipynb','requirements.txt','README.md','PROTOCOL.md']:
            if (package/name).exists():z.write(package/name,'research_package/'+name)
    return dest

def infer_coughs(run_dir,wav_paths,answers,hr_quality_ok=False,device='cpu'):
    """Future web integration adapter for ALREADY SEGMENTED cough WAVs.

    It does not detect coughs in long browser recordings or estimate heart rate.
    Only load local trusted model bundles. joblib is not safe for untrusted files.
    """
    run_dir=Path(run_dir);selection=json.loads((run_dir/'selection.json').read_text())
    hr=answers.get('heart_rate');valid_hr=hr is not None and np.isfinite(float(hr)) and 30<=float(hr)<=220 and hr_quality_ok
    name=selection['primary'] if valid_hr else selection['fallback_without_hr']
    b=load_bundle(run_dir/'models'/name,device);meta=json.loads((run_dir/'models'/name/'metadata.json').read_text())
    cfg={'sample_rate':meta['sample_rate'],'window_seconds':meta['window_seconds'],'deep_batch_size':64}
    if not wav_paths:raise ValueError('At least one segmented cough WAV is required')
    # Reject multi-second inputs instead of silently classifying the first 0.5 s.
    for path in wav_paths:
        with wave.open(str(path),'rb') as w:
            if w.getnframes()/w.getframerate()>.55:raise ValueError('Segment cough events first; this adapter accepts <=0.55-second WAVs')
    waves=np.stack([read_audio(p,cfg) for p in wav_paths]);features=[acoustic_features(x,cfg['sample_rate']) for x in waves]
    row={**answers,**dict(zip(features[0][1],np.mean([f[0] for f in features],axis=0)))}
    for col in b['columns']:row.setdefault(col,np.nan)
    if not valid_hr:row['heart_rate']=np.nan
    table=pd.DataFrame([row],index=['inference']);index=pd.DataFrame({'participant':['inference']*len(waves),'filename':[Path(x).name for x in wav_paths]})
    probability=float(predict_bundle(b,table,['inference'],waves,index,cfg)[0])
    return {'model':name,'screening_score':probability,'threshold':float(b['threshold']),
        'screening_result':'screen_positive' if probability>=b['threshold'] else 'screen_negative',
        'hr_used':bool(valid_hr),'n_coughs':len(wav_paths),'not_a_diagnosis':True,
        'hr_camera_reference_validated':False,'model_version':VERSION}
