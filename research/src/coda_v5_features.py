"""Label-free audio features and pinned local assets for V5."""
from pathlib import Path
import gc, hashlib, json, math, time, wave
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
from . import coda_research as cr
from .coda_v4 import atomic_json, atomic_model, digest


def read_physical(path,sample_rate=16000):
    with wave.open(str(path),'rb') as w:
        if w.getnchannels()!=1 or w.getsampwidth()!=2:raise ValueError('Expected mono PCM16')
        sr=w.getframerate();x=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32)/32768.
    # Match author LibROSA resampling; retain physical amplitude and true length.
    if sr!=sample_rate:
        import librosa
        x=librosa.resample(x,orig_sr=sr,target_sr=sample_rate)
    if len(x)<400 or not np.isfinite(x).all():raise ValueError('Invalid/too short WAV')
    return x.astype(np.float32)


def paper_features(x,sr=16000):
    """261 LibROSA descriptors, functional-major ordering, author-code parity."""
    import librosa
    opts=dict(n_fft=2048,win_length=512,hop_length=256)
    s=np.abs(librosa.stft(x,**opts))
    a=librosa.feature.mfcc(y=x,sr=sr,n_mfcc=13,**opts)
    b=librosa.feature.chroma_stft(y=x,sr=sr,**opts)
    spectral=[librosa.feature.spectral_centroid(S=s,sr=sr,win_length=512,hop_length=256),
              librosa.feature.spectral_bandwidth(S=s,sr=sr,win_length=512,hop_length=256),
              librosa.feature.spectral_rolloff(S=s,sr=sr,win_length=512,hop_length=256),
              librosa.feature.spectral_flatness(S=s,win_length=512,hop_length=256)]
    tracks=np.vstack([a,b,*spectral])
    values=[tracks.mean(axis=1),tracks.std(axis=1),skew(tracks,axis=1,bias=False),
            kurtosis(tracks,axis=1,fisher=True,bias=False)]
    values += [np.percentile(tracks,q,axis=1) for q in [10,25,50,75,90]]
    # Keep undefined moments missing; imputation is fitted only inside train.
    result=np.concatenate(values).astype(np.float32)
    result[~np.isfinite(result)]=np.nan
    assert result.shape==(261,)
    return result


def encode_clinical(participants):
    from .coda_v5_models import CLINICAL
    t=participants.set_index('participant').sort_index().copy()
    numeric=cr.NUMERIC
    for col in CLINICAL+['heart_rate']:
        if col in numeric:t[col]=pd.to_numeric(t[col].astype(str).str.replace(',','.',regex=False),errors='coerce')
        else:
            mapping=({'male':1.,'m':1.,'female':0.,'f':0.} if col=='sex' else
                     {'yes':1.,'y':1.,'true':1.,'1':1.,'1.0':1.,'no':0.,'n':0.,'false':0.,'0':0.,'0.0':0.})
            t[col]=t[col].astype(str).str.strip().str.lower().map(mapping)
    return t


def verify_assets(package):
    package=Path(package)
    record=json.loads((package/'assets_v5.json').read_text())
    for a in record['assets']:
        if cr.sha256(package/a['path'])!=a['sha256']:raise ValueError('Asset checksum mismatch: '+a['path'])
    return record


def gpu_guard(cfg,phase):
    import torch
    if not torch.cuda.is_available():return {'phase':phase,'used_gb':0.,'cuda':False}
    free,total=torch.cuda.mem_get_info(0);used=total-free
    if used>cfg['vram_limit_gb']*1e9:
        raise RuntimeError(f'GPU total usage {used/1e9:.2f} GB exceeds 45 GB policy at {phase}. Close other GPU workloads.')
    return {'phase':phase,'used_gb':used/1e9,'total_gb':total/1e9,'cuda':True,
            'torch_reserved_gb':torch.cuda.memory_reserved()/1e9,
            'torch_peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9}


def load_beats(cfg,device=None):
    import torch
    from .vendor_beats.BEATs import BEATs,BEATsConfig
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    path=Path(cfg['package_root'])/cfg['beats_checkpoint']
    # Pinned state dict with primitive configuration, never an arbitrary model object.
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    config=BEATsConfig(ckpt['cfg'])
    if config.finetuned_model:raise ValueError('Expected pretrained encoder, not AudioSet classifier output')
    net=BEATs(config);net.load_state_dict(ckpt['model'],strict=True)
    net.eval().requires_grad_(False)
    return net.to(device)


def embed_batch(net,waves,cfg,device):
    import torch
    # Padding to a common 0.5s window is fixed per recording, independent of label.
    # Preserve physical PCM amplitude; no V1 per-clip standardization.
    a=np.stack([np.pad(x[:8000],(0,max(0,8000-len(x)))) for x in waves])
    with torch.inference_mode():
        source=torch.from_numpy(a).to(device)
        amp=cfg.get('beats_precision')=='float16' and device=='cuda'
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):
            features,_=net.extract_features(source)
        out=features.float().mean(dim=1).cpu().numpy()
    if not np.isfinite(out).all():raise ValueError('Non-finite BEATs output')
    return out.astype(np.float32)


def choose_batch(net,waves,cfg,device):
    import torch
    if device!='cuda':return cfg['beats_batch_size'],[]
    trials=[];batch=16
    while batch<=cfg['beats_max_batch_size']:
        sample=[waves[i%len(waves)] for i in range(batch)]
        try:
            gpu_guard(cfg,'batch benchmark')
            torch.cuda.synchronize();stamp=time.perf_counter()
            embed_batch(net,sample,cfg,device)
            torch.cuda.synchronize();seconds=time.perf_counter()-stamp
            usage=gpu_guard(cfg,'batch benchmark complete')
            trials.append(dict(batch=batch,seconds=seconds,coughs_per_second=batch/seconds,**usage))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache();break
        if usage['used_gb']>(cfg['vram_limit_gb']-cfg['vram_headroom_gb']):break
        batch*=2
    if not trials:raise RuntimeError('No feasible encoder GPU batch; close other GPU workloads.')
    selected=max(trials,key=lambda r:r['coughs_per_second'])['batch']
    print('Measured BEATs batch:',selected,'| benchmark batches:',[r['batch'] for r in trials],flush=True)
    return selected,trials


def prepare_features(participants,manifest,cfg,include_beats=True):
    import joblib
    package=Path(cfg['package_root']);out=Path(cfg['output_dir'])
    t=encode_clinical(participants);ordered=manifest.sort_values('filename').reset_index(drop=True)
    rows=list(ordered.itertuples());paths=[Path(cfg['data_root'])/r.path for r in rows]
    import importlib.metadata as md
    provenance={'manifest':cr.sha256(cfg['manifest']),'features_code':cr.sha256(__file__),
                'versions':{p:md.version(p) for p in ['librosa','numpy','scipy','soxr']}}
    fp=digest(provenance);cache=package/'cache_v5'/fp[:16];cache.mkdir(parents=True,exist_ok=True)
    paperpath=cache/'paper.npz';papermeta=cache/'paper.json'
    if paperpath.exists() and papermeta.exists():
        if cr.sha256(paperpath)!=json.loads(papermeta.read_text())['sha256']:raise ValueError('Paper cache checksum mismatch')
        paper=np.load(paperpath)['features']
    else:
        def extract(path):return paper_features(read_physical(path))
        features=[];start=time.perf_counter()
        # Warm LibROSA's lazy imports/Numba once before worker threads start.
        extract(paths[0])
        with ThreadPoolExecutor(max_workers=cfg['feature_workers']) as pool:
            for i,f in enumerate(pool.map(extract,paths),1):
                features.append(f)
                if i%500==0 or i==len(paths):print(f'LibROSA {i}/{len(paths)} | {time.perf_counter()-start:.1f}s',flush=True)
        paper=np.stack(features)
        tmp=paperpath.with_suffix('.part')
        with tmp.open('wb') as f:np.savez_compressed(f,features=paper)
        tmp.replace(paperpath);atomic_json(papermeta,dict(provenance,sha256=cr.sha256(paperpath)))
    if paper.shape!=(len(rows),261):raise ValueError('Feature count mismatch')
    features={'paper':paper};metadata={'paper':json.loads(papermeta.read_text()),'frozen_encoder_label_free':True}
    if include_beats:
        import torch
        device='cuda' if torch.cuda.is_available() else 'cpu'
        if cfg['require_cuda_for_full'] and device!='cuda':raise RuntimeError('Full BEATs extraction requires CUDA; use MODE=preflight on laptop.')
        enc_hash=cr.sha256(package/cfg['beats_checkpoint'])
        efp=digest({'paper_fp':fp,'checkpoint':enc_hash,'precision':cfg['beats_precision'],'window':8000,
                    'encoder_assets':cr.sha256(package/'assets_v5.json'),
                    'torch':md.version('torch'),'torchaudio':md.version('torchaudio')})
        epath=cache/('beats_'+efp[:16]+'.npz');emeta=epath.with_suffix('.json')
        if epath.exists() and emeta.exists():
            if cr.sha256(epath)!=json.loads(emeta.read_text())['sha256']:raise ValueError('BEATs cache mismatch')
            embeddings=np.load(epath)['features']
        else:
            gpu_guard(cfg,'before BEATs');net=load_beats(cfg,device)
            bs,benchmark=choose_batch(net,[read_physical(p) for p in paths[:8]],cfg,device)
            embeddings=np.empty((len(rows),768),np.float32);logs=[]
            # Each batch is a resumable chunk with fingerprinted provenance.
            chunks=cache/('beats_chunks_'+efp[:16]);chunks.mkdir(exist_ok=True)
            start=0
            while start<len(rows):
                stop=min(start+bs,len(rows));cp=chunks/f'{start:06d}_{stop:06d}.npz'
                cm=cp.with_suffix('.json')
                if cp.exists() and cm.exists():
                    if cr.sha256(cp)!=json.loads(cm.read_text())['sha256']:raise ValueError('Encoder chunk mismatch')
                    values=np.load(cp)['features']
                else:
                    stamp=time.perf_counter();waves=[read_physical(p) for p in paths[start:stop]]
                    try:
                        gpu_guard(cfg,'before encoder batch');values=embed_batch(net,waves,cfg,device)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if bs<=1:raise
                        bs=max(1,bs//2);print('BEATs OOM: retry batch',bs,flush=True);continue
                    with cp.with_suffix('.part').open('wb') as f:np.savez_compressed(f,features=values)
                    cp.with_suffix('.part').replace(cp);atomic_json(cm,{'sha256':cr.sha256(cp),'start':start,'stop':stop})
                    logs.append(dict(start=start,stop=stop,batch_size=bs,seconds=time.perf_counter()-stamp,**gpu_guard(cfg,'encoder batch')))
                if values.shape!=(stop-start,768):raise ValueError('Encoder chunk shape mismatch')
                embeddings[start:stop]=values;start=stop
                if start%512<bs or start==len(rows):print(f'BEATs {start}/{len(rows)} | batch={bs}',flush=True)
            del net;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
            with epath.with_suffix('.part').open('wb') as f:np.savez_compressed(f,features=embeddings)
            epath.with_suffix('.part').replace(epath)
            atomic_json(emeta,{'checkpoint_sha256':enc_hash,'sha256':cr.sha256(epath),'embedding_dim':768,
                              'precision':cfg['beats_precision'],'batch_log':logs,'batch_benchmark':benchmark})
        if embeddings.shape!=(len(rows),768) or not np.isfinite(embeddings).all():raise ValueError('BEATs feature validation failed')
        features['beats']=embeddings;metadata['beats']=json.loads(emeta.read_text())
    for name,values in features.items():
        bags={pid:values[np.flatnonzero(ordered.participant.to_numpy()==pid)] for pid in t.index}
        if not all(len(b)>0 for b in bags.values()):raise ValueError('Empty participant bag')
        t['_bag_'+name]=pd.Series(bags,dtype=object)
    atomic_model(out/'participant_features.joblib',t)
    atomic_json(out/'feature_provenance.json',metadata)
    return t
