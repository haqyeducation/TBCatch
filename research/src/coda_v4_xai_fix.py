"""Canonical full-coalition prediction for V4 XAI; checkpoint-safe runtime fix."""
from pathlib import Path
import itertools,math
import numpy as np
import pandas as pd
from . import coda_v4 as v4
from . import coda_research as cr


def _shapley(bundle,table,pid,background):
    keys=list(bundle['groups']);n=len(keys)
    coalitions=list(itertools.product((0,1),repeat=n));frames=[]
    for bits in coalitions:
        t=background.copy()
        for present,group in zip(bits,keys):
            if present:
                for col in bundle['groups'][group]:
                    t[col]=pd.Series([table.at[pid,col]]*len(t),index=t.index,dtype=table[col].dtype)
        frames.append(t)
    combined=pd.concat(frames)
    raw=np.asarray(v4.predict_raw(bundle['model'],combined),dtype=float)
    single_raw=float(v4.predict_raw(bundle['model'],table.loc[[pid]])[0])
    if not np.isfinite(raw).all() or not np.isfinite(single_raw):
        raise RuntimeError(f'Nonfinite XAI predictions for {pid}; keep checkpoints for inspection.')
    full_raw=raw.reshape(len(coalitions),len(background))[-1]
    batch_raw_delta=float(np.max(np.abs(full_raw-single_raw)))
    batch_probability=float(np.clip(bundle['calibrator'].predict(full_raw),0,1).mean())
    rowwise=batch_raw_delta>1e-10
    if rowwise:
        # XAI must evaluate f(x) consistently when prediction depends on batch shape.
        print(f'XAI {pid}: batch delta={batch_raw_delta:.3g}; memakai inferensi per baris.',flush=True)
        raw=np.asarray([v4.predict_raw(bundle['model'],combined.iloc[[i]])[0]
                        for i in range(len(combined))],dtype=float)
        full_raw=raw.reshape(len(coalitions),len(background))[-1]
    raw_delta=float(np.max(np.abs(full_raw-single_raw)))
    if not np.isfinite(raw).all() or not np.isfinite(single_raw) or raw_delta>1e-10:
        raise RuntimeError(f'XAI input/model mismatch for {pid}: full vs single raw delta={raw_delta:.3g}. '
                           'Keep checkpoints; inspect model/groups instead of loosening completeness.')
    calibrated=np.clip(bundle['calibrator'].predict(raw),0,1)
    values=dict(zip(coalitions,calibrated.reshape(len(coalitions),len(background)).mean(axis=1)))
    probability=float(np.clip(bundle['calibrator'].predict([single_raw]),0,1)[0])
    # All-present coalition equals f(x); tiny BLAS differences can be amplified by isotonic.
    values[(1,)*n]=probability
    phi={}
    for j,group in enumerate(keys):
        phi[group]=0.
        for bits,value in values.items():
            if bits[j]:continue
            k=sum(bits);other=list(bits);other[j]=1
            phi[group]+=math.factorial(k)*math.factorial(n-k-1)/math.factorial(n)*(values[tuple(other)]-value)
    base=float(values[(0,)*n]);residual=probability-base-sum(phi.values())
    assert abs(residual)<1e-8, f'XAI completeness failed for {pid}: residual={residual}'
    return {'base_probability':base,'probability':probability,'completeness_residual':residual,
        'full_coalition_batch_probability':batch_probability,'full_vs_single_raw_delta':raw_delta,
        'batch_full_vs_single_raw_delta':batch_raw_delta,'rowwise_fallback_used':rowwise,
        'full_vs_single_calibrated_delta':abs(batch_probability-probability),**phi}


def apply(cfg):
    out=Path(cfg['output_dir'])
    if (out/'RUN_COMPLETE.json').exists():return False
    core=Path(cfg['package_root'])/'src/coda_v4.py'
    if cr.sha256(core)!=cfg['source_hashes']['src/coda_v4.py']:
        raise RuntimeError('Original V4 source changed; restore it before resuming.')
    v4._shapley=_shapley
    source=out/'runtime_sources/coda_v4_xai_fix.py';source.parent.mkdir(parents=True,exist_ok=True)
    source.write_bytes(Path(__file__).read_bytes())
    v4.atomic_json(out/'XAI_FIX_AUDIT.json',{'fix':'canonical_full_coalition_v2_rowwise_fallback','source_sha256':cr.sha256(__file__),
        'training_models_thresholds_unchanged':True,'raw_prediction_agreement_required':1e-10,
        'completeness_tolerance_unchanged':1e-8,
        'notice':'XAI falls back to rowwise inference on batch discrepancy, then enforces raw agreement. Training/evaluation scoring and completed folds are preserved; batch discrepancy remains an audit concern.'})
    print('XAI fix aktif; checkpoint/model/cutoff lama tetap digunakan.',flush=True)
    return True
