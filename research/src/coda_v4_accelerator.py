"""Optional V4 runtime acceleration; protected V4 sources/config stay unchanged.

No changes to candidate grid, folds, classifiers, seeds, calibration or XAI.
Memoize exact deterministic CPU fits and audio preprocessing; optionally import
verified, equivalent V3 legacy OOF predictions. Never cache neural training.
"""
from pathlib import Path
from collections import OrderedDict
import copy, hashlib, io, json, pickle, threading, time, zipfile
import importlib.metadata as md
import platform
import joblib
import numpy as np
import pandas as pd
from . import coda_v4 as v4
from . import coda_v4_models as engines
from . import coda_research as cr

_ACTIVE=None


def feature_digest(table):
    h=hashlib.sha256(json.dumps([(str(c),str(table[c].dtype)) for c in table],sort_keys=True).encode())
    h.update(pd.util.hash_pandas_object(table,index=True).to_numpy(dtype='<u8').tobytes())
    return h.hexdigest()


class BoundedCache:
    def __init__(self,max_bytes,max_entries):
        self.data=OrderedDict();self.bytes=0;self.max_bytes=max_bytes;self.max_entries=max_entries
        self.hits=0;self.misses=0

    def get(self,key):
        if key not in self.data:self.misses+=1;return None
        value,size=self.data.pop(key);self.data[key]=(value,size);self.hits+=1;return value

    def put(self,key,value):
        size=len(pickle.dumps(value,protocol=4))
        if size>self.max_bytes:return
        if key in self.data:self.bytes-=self.data.pop(key)[1]
        while self.data and (self.bytes+size>self.max_bytes or len(self.data)>=self.max_entries):
            _,(_,removed)=self.data.popitem(last=False);self.bytes-=removed
        self.data[key]=(value,size);self.bytes+=size


class Session:
    def __init__(self,cfg,heartbeat_seconds=30,verbose=True):
        self.cfg=cfg;self.out=Path(cfg['output_dir']);self.verbose=verbose;self.interval=heartbeat_seconds
        self.audio=BoundedCache(96*1024**2,64);self.models=BoundedCache(256*1024**2,128)
        self.original_fit=engines.fit_model;self.original_audio=engines._audio_preprocessor
        self.original_v4_fit=v4.fit_model;self.depth=0;self.calls=0;self.bank_report={};self.closed=False
        self.fit_records=[]

    def audio_preprocessor(self,x,candidate,seed,weights=None):
        key=joblib.hash((x,candidate.get('pca'),seed,weights))
        value=self.audio.get(key)
        if value is None:
            value=self.original_audio(x,candidate,seed,weights);self.audio.put(key,value)
        return value

    def fit_model(self,table,ids,candidate,cfg,seed):
        top=self.depth==0;self.depth+=1;self.calls+=1
        label=f"{candidate.get('id','base')} | {candidate['engine']}/{candidate['family']} | seed {seed}"
        stop=threading.Event();thread=None;start=time.perf_counter();cache_hit=False
        if top and self.verbose:
            print('Mulai fit:',label,flush=True)
            def heartbeat():
                while not stop.wait(self.interval):print(f'  Masih fitting {label} | {time.perf_counter()-start:.0f}s',flush=True)
            thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:
            reusable=candidate['engine'] in ['patient','clip']
            key=None;value=None
            if reusable:
                groups=engines.groups_for(table,candidate);cols=sum(groups.values(),[])
                options={k:candidate.get(k) for k in ['engine','family','C','balanced','pca','params']}
                key=joblib.hash((table.loc[list(ids),cols+['tb_status']],groups,options,seed,
                    cfg['cpu_threads'],cfg['protocol_fingerprint']))
                value=self.models.get(key)
            if value is None:
                value=self.original_fit(table,ids,candidate,cfg,seed)
                if reusable:self.models.put(key,value)
            else:
                cache_hit=True
                # Estimators are read-only after fitting; use a fresh metadata
                # container so source candidate IDs cannot leak into exports.
                value=dict(value,candidate=copy.deepcopy(candidate),fit_ids=list(ids))
            return value
        finally:
            self.depth-=1;stop.set()
            if thread:thread.join(timeout=1)
            if top:self.fit_records.append({'candidate':candidate.get('id'),'engine':candidate['engine'],
                'family':candidate['family'],'seed':seed,'seconds':time.perf_counter()-start,'exact_fit_cache_hit':cache_hit})
            if top and self.verbose:print(f'Fit selesai: {label} | {time.perf_counter()-start:.1f}s',flush=True)

    def flush(self):
        record={'addon_sha256':cr.sha256(__file__),'candidate_grid_unchanged':True,'training_backend_unchanged':True,
            'audio_cache_hits':self.audio.hits,'audio_cache_misses':self.audio.misses,
            'exact_fit_cache_hits':self.models.hits,'exact_fit_cache_misses':self.models.misses,
            'audio_cache_bytes':self.audio.bytes,'model_cache_bytes':self.models.bytes,
            'legacy_bank':self.bank_report,'notice':'Runtime cache only; no pruning, GPU conversion or changed fitting parameters.'}
        v4.atomic_json(self.out/'ACCELERATION_AUDIT.json',record)
        if self.fit_records:pd.DataFrame(self.fit_records).to_csv(self.out/'ACCELERATION_FIT_TIMINGS.csv',index=False)

    def close(self):
        global _ACTIVE
        if self.closed:return
        engines.fit_model=self.original_fit;engines._audio_preprocessor=self.original_audio;v4.fit_model=self.original_v4_fit
        self.flush();self.audio.data.clear();self.models.data.clear();self.closed=True;_ACTIVE=None


def import_legacy_bank(bank,table,splits,cfg):
    """Conservative equivalence checks; no import on environment/data mismatch."""
    bank=Path(bank)
    if not bank.exists():return {'status':'not_found','imported_candidates':0}
    imported=existing=0
    with zipfile.ZipFile(bank) as z:
        metadata=json.loads(z.read('metadata.json'))
        assert metadata['splits']==splits,'V3/V4 splits differ'
        for name,h in metadata['required_source_hashes'].items():assert cr.sha256(Path(cfg['package_root'])/name)==h,name
        for name,version in metadata['packages'].items():
            assert md.version(name)==version,f'{name}: current={md.version(name)}, V3={version}'
        assert platform.python_version()==metadata['python'],'V3 Python version differs'
        assert platform.platform()==metadata['platform'],'V3 operating system differs'
        assert cfg['seed']==metadata['seed'] and cfg['pca_components']==metadata['pca_components']
        assert cfg['cpu_threads']==metadata['cpu_threads']
        assert feature_digest(table[metadata['feature_columns']])==metadata['feature_digest'],'Legacy features changed'
        checks=metadata['checksums']
        # Validate all payloads before creating any checkpoint.
        for name,h in checks.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
        candidates=[c for c in v4.candidate_grid(cfg) if c['engine']=='legacy']
        plan=[]
        for fold in splits['folds']:
            folder=Path(cfg['output_dir'])/f'outer_{fold["outer_fold"]:02d}'
            if (folder/'FOLD_COMPLETE.json').exists():
                existing+=len(candidates);continue
            signature=v4.digest({'protocol':cfg['protocol_fingerprint'],'fold':fold})
            for candidate in candidates:
                dest=folder/'inner_checkpoints'/f'{candidate["id"]}.json'
                if dest.exists():existing+=1;continue
                oldid=candidate['v3_candidate']['id'];name=f'outer_{fold["outer_fold"]:02d}/{oldid}.json'
                saved=json.loads(z.read(name))
                assert saved['candidate']==candidate['v3_candidate']
                assert len(saved['predictions'])==len(fold['proper_train'])
                assert {r['participant'] for r in saved['predictions']}==set(fold['proper_train'])
                labels=table.tb_status
                assert all(r['y']==labels.loc[r['participant']] for r in saved['predictions'])
                for i,sp in enumerate(fold['inner_folds']):
                    assert {r['participant'] for r in saved['predictions'] if r['inner_fold']==i}==set(sp['validation'])
                plan.append((dest,name,signature,candidate))
        for dest,name,signature,candidate in plan:
                saved=json.loads(z.read(name))
                v4.atomic_json(dest,{'signature':signature,'candidate':candidate,'predictions':saved['predictions'],
                    'statistics':saved['statistics'],'fit_seconds':0.,'imported_equivalent_v3':True,
                    'source_checkpoint_sha256':checks[name],'original_v3_fit_seconds':saved.get('fit_seconds')})
                imported+=1
    return {'status':'verified','imported_candidates':imported,'existing_candidates_preserved':existing,
        'bank_sha256':cr.sha256(bank),'notice':'Equivalent V3 raw proper-training OOF only; no outer test used.'}


def activate(cfg,table=None,splits=None,bank=None,verbose=True,heartbeat_seconds=30):
    global _ACTIVE
    if _ACTIVE is not None:raise RuntimeError('Accelerator already active; close the previous session first.')
    if (Path(cfg['output_dir'])/'RUN_COMPLETE.json').exists():
        raise RuntimeError('Run already complete. Use original notebook to audit/export; acceleration is unnecessary.')
    # Must still match the exact protected V4 code/config that created the run.
    for name,h in cfg['source_hashes'].items():assert cr.sha256(Path(cfg['package_root'])/name)==h,name
    session=Session(cfg,heartbeat_seconds,verbose)
    if bank is not None and table is not None and splits is not None:
        try:session.bank_report=import_legacy_bank(bank,table,splits,cfg)
        except AssertionError as error:
            session.bank_report={'status':'rejected_before_import','reason':str(error),
                'notice':'Unmatched V3 cache not used further; normal V4 fitting remains available.'}
        if verbose:print('Legacy reuse:',session.bank_report,flush=True)
    engines.fit_model=session.fit_model;v4.fit_model=session.fit_model;engines._audio_preprocessor=session.audio_preprocessor
    _ACTIVE=session
    source=session.out/'runtime_sources/coda_v4_accelerator.py';source.parent.mkdir(parents=True,exist_ok=True)
    source.write_bytes(Path(__file__).read_bytes());session.flush()
    return session
