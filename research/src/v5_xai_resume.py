"""Postprocessing-only V5 repair. Original sources/predictor/checkpoints stay intact."""
from pathlib import Path
import importlib.metadata as md
import json
import joblib
import numpy as np
import pandas as pd
from . import coda_v5 as v5
from . import coda_v5_models as models
from . import coda_research as cr

_SHAPLEY = v5.shapley_groups
_RAW = models.predict_raw
_ARCHIVED = {}
VERSION = 'v5-xai-pointwise-1'


def _columns(bundle, groups):
    required = sum(bundle['groups'].values(), [])
    supplied = sum(groups.values(), [])
    if len(supplied) != len(set(supplied)) or set(supplied) != set(required):
        raise RuntimeError('XAI groups do not cover exactly the model inputs. Inspect groups; do not relax prediction tolerance.')
    return sorted(set(required))


def _key(row, columns):
    # IDs and dataframe dtypes are not model inputs. Clinical columns are numeric.
    values = tuple((c, np.ascontiguousarray(row[c]) if isinstance(row[c], np.ndarray)
                    else float(row[c])) for c in columns)
    return joblib.hash(values)


def _equal_inputs(actual, expected, columns):
    for c in columns:
        a, b = actual[c], expected[c]
        if isinstance(b, np.ndarray):
            equal = isinstance(a, np.ndarray) and np.array_equal(a, b, equal_nan=True)
        else:
            equal = (pd.isna(a) and pd.isna(b)) or a == b
        if not equal: raise RuntimeError(f'All-present XAI input differs from participant: {c}')


def stable_shapley(bundle, table, pid, bg, groups, cfg, seed=0, permutations=None):
    columns = _columns(bundle, groups)
    if len(bg) == 0: raise ValueError('XAI background is empty')
    root = bundle['model']
    context = bundle.setdefault('_v5_xai_runtime', {'cache': {}, 'checked': set(),
                                                  'pointwise': False, 'probes': 0})
    cache = context['cache']; patient = table.loc[pid]; patient_key = _key(patient, columns)

    def one(frame):
        key = _key(frame.iloc[0], columns)
        if key not in cache:
            p = float(_RAW(root, frame, cfg)[0])
            if not np.isfinite(p) or not 0 <= p <= 1: raise RuntimeError(f'Invalid XAI prediction for {pid}')
            cache[key] = p
        return cache[key]

    single_raw = one(table.loc[[pid]])
    if patient_key not in context['checked']:
        repeated = float(_RAW(root, table.loc[[pid]], cfg)[0])
        if not np.isfinite(repeated) or abs(repeated-single_raw) > 1e-10:
            raise RuntimeError(f'Pointwise predictor is not repeatable for {pid}: delta={abs(repeated-single_raw):.6g}. Keep checkpoints for inspection.')
        context['checked'].add(patient_key)
    diagnostic = {'batch_probe_performed': False, 'original_batch_raw_delta': np.nan,
                  'original_batch_calibrated_delta': np.nan, 'pointwise_cache_misses': 0}
    before = len(cache)

    def xai_predict(model, frame, settings=None):
        if model is not root: return _RAW(model, frame, settings)
        if len(frame) == 1: return np.array([one(frame)])
        # Original Shapley orders the all-present coalition last, in both modes.
        full = frame.iloc[-len(bg):]
        for _, row in full.iterrows(): _equal_inputs(row, patient, columns)
        if not context['pointwise']:
            batch = np.asarray(_RAW(root, frame, settings), dtype=float)
            if len(batch) != len(frame) or not np.isfinite(batch).all(): raise RuntimeError('Invalid batched XAI prediction')
            raw_delta = float(np.max(abs(batch[-len(bg):]-single_raw)))
            cal_delta = abs(float(bundle['calibrator'].predict(batch[-len(bg):]).mean())-
                            float(bundle['calibrator'].predict([single_raw])[0]))
            diagnostic.update(batch_probe_performed=True, original_batch_raw_delta=raw_delta,
                              original_batch_calibrated_delta=cal_delta)
            context['probes'] += 1
            if raw_delta <= 1e-10 and cal_delta <= 1e-10: return batch
            context['pointwise'] = True
            context['trigger_raw_delta'] = raw_delta
            print(f'XAI {pid} | {root["candidate"]["id"]}: batch delta={raw_delta:.6g}; inferensi per baris + cache.', flush=True)
        return np.array([one(frame.iloc[[i]]) for i in range(len(frame))])

    # Restrict the predictor override to this XAI call; even exceptions restore it.
    if models.predict_raw is not _RAW: raise RuntimeError('Another predictor patch is active; restart the kernel before resuming XAI.')
    try:
        models.predict_raw = xai_predict
        result = _SHAPLEY(bundle, table, pid, bg, groups, cfg, seed, permutations)
    finally:
        models.predict_raw = _RAW
    diagnostic['pointwise_cache_misses'] = len(cache)-before
    result.update(diagnostic, rowwise_fallback_used=context['pointwise'],
                  fallback_trigger_raw_delta=context.get('trigger_raw_delta', 0.),
                  pointwise_full_raw_delta=result['raw_batch_abs_delta'])
    if result['pointwise_full_raw_delta'] > 1e-10 or abs(result['completeness_residual']) > 1e-8:
        raise RuntimeError('Pointwise XAI consistency/completeness failed; keep checkpoints for inspection.')
    archived = _ARCHIVED.get((root['candidate']['id'], str(pid)))
    if archived:
        probability = result['probability']
        result.update(archived_test_raw_delta=abs(single_raw-archived['raw_probability']),
                      archived_test_calibrated_delta=abs(probability-archived['probability']),
                      archived_youden_decision_changed=int((probability >= archived['threshold']) != bool(archived['prediction'])))
    return result


def _protected(cfg, splits):
    out = Path(cfg['output_dir']); protected = {}
    for fold in splits['folds']:
        folder = out/f'outer_{fold["outer_fold"]:02d}'
        path = folder/'TRAIN_EVAL_COMPLETE.json'
        if not path.exists(): raise RuntimeError(f'Training/evaluation not complete: {folder.name}. This notebook does not train models.')
        record = json.loads(path.read_text())
        if record['protocol'] != cfg['protocol_fingerprint']: raise RuntimeError('Completed training fingerprint differs')
        # Validate the original recorded checksums, not freshly accepted values.
        for rel, expected in record['checksums'].items():
            p = folder/rel
            if cr.sha256(p) != expected: raise RuntimeError('Training/evaluation artifact changed: '+str(p))
            protected[p.relative_to(out).as_posix()] = expected
        protected[path.relative_to(out).as_posix()] = cr.sha256(path)
    return protected


def apply(cfg, splits=None):
    package = Path(cfg['package_root']); out = Path(cfg['output_dir'])
    for rel, expected in cfg['source_hashes'].items():
        if cr.sha256(package/rel) != expected:
            raise RuntimeError('Original V5 source changed: '+rel+'. Restore the original package before applying this runtime-only fix.')
    for name, expected in cfg.get('versions', {}).items():
        if md.version(name) != expected: raise RuntimeError(f'Library changed: {name}; use the original training kernel.')
    import torch
    if str(cfg.get('tabpfn_device','cpu')).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('This run uses CUDA. Resume on the original training PC; no silent CPU backend change.')
    cr.seed_everything(cfg['seed'])
    torch.set_num_threads(cfg['cpu_threads'])
    if torch.cuda.is_available():
        from .coda_v5_features import gpu_guard
        gpu_guard(cfg,'XAI resume')
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.9,(cfg['vram_limit_gb']-cfg['vram_headroom_gb'])*1e9/total),0)
    splits = splits or json.loads((out/'nested_splits.json').read_text())
    protected = _protected(cfg, splits)
    source_hash = cr.sha256(__file__)
    destination = out/'runtime_sources'/f'v5_xai_resume_{source_hash[:12]}.py'
    destination.parent.mkdir(exist_ok=True)
    if destination.exists() and cr.sha256(destination) != source_hash: raise RuntimeError('Runtime patch copy changed')
    destination.write_bytes(Path(__file__).read_bytes())
    _ARCHIVED.clear()
    for fold in splits['folds']:
        folder = out/f'outer_{fold["outer_fold"]:02d}'
        selection = json.loads((folder/'selection.json').read_text())
        winners = dict(selection['winners'])
        winners['selected_full'] = winners[selection['primary_full_output']]
        predictions = pd.read_csv(folder/'outer_test_predictions.csv', float_precision='round_trip')
        for r in predictions[predictions.policy == 'youden'].to_dict('records'):
            _ARCHIVED[(winners[r['combination']], str(r['participant']))] = r
    completed = [f['outer_fold'] for f in splits['folds'] if
                 (out/f'outer_{f["outer_fold"]:02d}'/'POSTPROCESS_COMPLETE.json').exists()]
    audit_path=out/'V5_XAI_RUNTIME_FIX.json'
    previous=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    if previous and previous['source_sha256']!=source_hash:
        raise RuntimeError('A different XAI runtime patch was recorded. Preserve its audit before changing patch versions.')
    v5.atomic_json(out/'V5_XAI_RUNTIME_FIX.json', dict(version=VERSION, source_sha256=source_hash,
        protected_training_artifacts=protected,
        completed_postprocess_folds_before_fix=previous.get('completed_postprocess_folds_before_fix',completed),
        applications=int(previous.get('applications',0))+1,
        original_sources_unchanged=True, training_prediction_function_unchanged_outside_XAI=True,
        raw_pointwise_agreement=1e-10, completeness_tolerance=1e-8,
        explanation='Fallback explains pointwise inference. Original batch discrepancy and available archived-test deltas are retained; mathematical completeness is not clinical validation.'))
    if v5.shapley_groups not in (_SHAPLEY, stable_shapley): raise RuntimeError('Another XAI patch is active; restart the kernel.')
    v5.shapley_groups = stable_shapley
    print('V5 XAI runtime fix aktif. Semua training/model/cutoff/prediksi lama diverifikasi; tidak ada training ulang.', flush=True)
    return protected


def verify_unchanged(cfg, protected):
    out = Path(cfg['output_dir'])
    for rel, expected in protected.items():
        if cr.sha256(out/rel) != expected: raise RuntimeError('Protected training artifact changed: '+rel)
    v5.atomic_json(out/'V5_XAI_PROTECTION_VERIFIED.json', dict(passed=True, count=len(protected),
                  source_sha256=cr.sha256(__file__), training_unchanged=True))
    return True


def summarize(cfg):
    out=Path(cfg['output_dir']);frame=pd.read_csv(out/'nested_grouped_xai.csv')
    result={'version':VERSION,'sampled_grouped_rows':len(frame),
            'sampled_unique_participants':int(frame.participant.nunique()),
            'completeness_max_abs':float(frame.completeness_residual.abs().max()),
            'notice':'Deltas concern the sampled XAI cases, not a replay of every test participant. Older completed postprocessing folds may lack new diagnostic columns.'}
    for name in ['original_batch_raw_delta','original_batch_calibrated_delta',
                 'archived_test_raw_delta','archived_test_calibrated_delta']:
        if name in frame:result[name+'_max']=frame[name].max()
    if 'rowwise_fallback_used' in frame:result['rowwise_fallback_rows']=int(frame.rowwise_fallback_used.fillna(False).sum())
    if 'archived_youden_decision_changed' in frame:
        result['sampled_archived_youden_decisions_changed']=int(frame.archived_youden_decision_changed.fillna(0).sum())
    v5.atomic_json(out/'V5_XAI_DIAGNOSTIC_SUMMARY.json',result)
    return result


def load_run(package, run_name):
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,70}',run_name): raise ValueError('Invalid run name')
    package = Path(package).resolve(); out = package/'runs'/run_name
    cfg = json.loads((out/'resolved_config.json').read_text())
    cfg.update(package_root=str(package), output_dir=str(out))
    splits = json.loads((out/'nested_splits.json').read_text())
    protected = apply(cfg, splits)
    table = joblib.load(out/'participant_features.joblib')
    cohort = pd.read_csv(out/'cohort.csv').set_index('participant')
    expected = [pid for f in splits['folds'] for pid in f['test']]
    if not table.index.is_unique or set(table.index) != set(expected): raise RuntimeError('Feature-table participant mismatch')
    if not np.array_equal(table.tb_status.to_numpy(),cohort.loc[table.index,'tb_status'].to_numpy()): raise RuntimeError('Feature-table labels differ from cohort')
    return cfg, table, splits, protected
