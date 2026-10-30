"""Synthetic end-to-end and real encoder/parity checks; never trains TB models."""
from pathlib import Path
import ast, copy, io, json, sys, tempfile
import numpy as np
import pandas as pd
import joblib, torch
from src import coda_research as cr
from src import coda_v5 as v5
from src import coda_v5_models as m
from src import coda_v5_features as f


def main():
    root=Path(__file__).resolve().parent
    tested=[root/'config_v5.json',*sorted((root/'src').glob('coda_v5*.py'))]
    source_hashes={p.relative_to(root).as_posix():cr.sha256(p) for p in tested}
    cfg=cr.load_config(root);cfg.update(json.loads((root/'config_v5.json').read_text()))
    cfg.update(package_root=str(root),outer_folds=2,inner_folds=2,cpu_threads=2,
        bootstrap_repeats=20,tabpfn_device='cpu',tabpfn_estimators=1,tabpfn_prediction_chunk=256,
        require_cuda_for_full=False,protocol_fingerprint='SYNTHETIC_V5',feature_workers=1,
        xai_per_outer_fold=1,xai_background=2,individual_xai_permutations=2)
    cfg['source_hashes']=source_hashes
    cfg['catboost_trials']=[dict(depth=2,iterations=8,learning_rate=.1,l2_leaf_reg=3)]*3
    torch.set_num_threads(2)
    rng=np.random.default_rng(42);n=80;ids=[f'SYNTH_{i:03d}' for i in range(n)]
    y=np.arange(n)%2;z=y+rng.normal(size=n)*.8;t=pd.DataFrame(index=ids)
    t['tb_status']=y;t['Country']='SYNTH'
    for col in m.CLINICAL:t[col]=z+rng.normal(size=n) if col in cr.NUMERIC else rng.integers(0,2,n).astype(float)
    t['heart_rate']=80+8*z;t.loc[ids[:3],'heart_rate']=np.nan
    for rep in ['paper','beats']:
        t['_bag_'+rep]=pd.Series([rng.normal(size=(int(rng.integers(2,5)),8))+zz for zz in z],index=ids,dtype=object)
    t['tb_prior_Pul']=np.nan  # An all-missing training column must retain its position.
    grid=m.candidate_grid(cfg);assert len(grid)==44
    assert all(set(m.CLINICAL+['heart_rate'])==set(sum(m.groups_for(t,c).values(),[])) for c in grid if c['output']=='clinical_full')
    f.verify_assets(root)
    # Actual pretrained BEATs, not a mocked encoder. 0.5-second cough shape works.
    net=f.load_beats(cfg,'cpu');wave=rng.normal(size=8000).astype(np.float32)*.02
    embedding=f.embed_batch(net,[wave],cfg,'cpu');assert embedding.shape==(1,768) and np.isfinite(embedding).all()
    together=f.embed_batch(net,[wave,wave],cfg,'cpu');assert np.allclose(embedding,together[:1],rtol=1e-4,atol=1e-5)
    del net
    # Compare formula implementation against the downloaded author's function,
    # without importing or executing their experiment driver.
    reference=root/'references/sensors_author/LR_LR_CV.py'
    parity=None
    if reference.exists():
        import librosa
        from scipy.stats import skew,kurtosis
        tree=ast.parse(reference.read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='feats_extract')
        node.returns=None
        for arg in node.args.args:arg.annotation=None
        node.args.defaults=[ast.Constant(13)]
        namespace=dict(np=np,librosa=librosa,skew=skew,kurtosis=kurtosis,N_MFCC=13)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'author_function','exec'),namespace)
        expected=namespace['feats_extract'](wave,16000)
        actual=f.paper_features(wave);assert np.allclose(actual,expected,equal_nan=True,rtol=1e-6,atol=1e-6)
        parity=float(np.nanmax(abs(actual-expected)))
    with tempfile.TemporaryDirectory(prefix='coda_v5_test_') as temp:
        cfg['output_dir']=temp;splits=v5.nested_splits(t.reset_index(names='participant'),cfg)
        first=splits['folds'][0]
        # Exercise every model family with actual local TabPFN weights.
        for family in ['lr','cb','tabpfn','blend']:
            c=next(c for c in grid if c['family']==family and c['output'] in ['beats_full_lr','beats_full_cb','beats_full_tabpfn','ensemble_full'])
            model=m.fit_model(t,first['proper_train'],c,cfg,42)
            a=m.predict_raw(model,t.loc[first['test']],cfg);assert np.isfinite(a).all()
            m.release(model);stream=io.BytesIO();joblib.dump(model,stream);stream.seek(0)
            loaded=joblib.load(stream);b=m.predict_raw(loaded,t.loc[first['test']],cfg)
            assert np.allclose(a,b,atol=1e-6),(family,float(np.max(abs(a-b))))
            reordered=t.loc[first['test']].copy()
            reordered['_bag_beats']=pd.Series([x[::-1].copy() for x in reordered._bag_beats],index=reordered.index,dtype=object)
            assert np.allclose(b,m.predict_raw(loaded,reordered,cfg),atol=1e-6)
            from sklearn.isotonic import IsotonicRegression
            bundle=dict(model=loaded,groups=loaded['groups'],calibrator=IsotonicRegression(out_of_bounds='clip').fit([0,1],[0,1]))
            explanation=v5.shapley_groups(bundle,t,first['test'][0],t.loc[first['proper_train'][:2]],loaded['groups'],cfg)
            assert abs(explanation['completeness_residual'])<1e-8
            m.release(loaded)
        v5.run_nested_cv(t,splits,cfg)
        metrics,pred,summary,pooled,paired=v5.aggregate_results(splits,cfg)
        assert len(pred)==n*(len(m.OUTPUTS)+1)*2 and len(metrics)==2*(len(m.OUTPUTS)+1)*2
        # Saved training/test predictions remain unchanged on a full resume.
        hashes={p.as_posix():cr.sha256(p) for p in Path(temp).glob('outer_*/models/*/bundle.joblib')}
        v5.run_nested_cv(t,splits,cfg)
        assert hashes=={p.as_posix():cr.sha256(p) for p in Path(temp).glob('outer_*/models/*/bundle.joblib')}
        # Fresh inner search cannot depend on outer-test or calibration labels/features.
        changed=t.copy();held=first['test']+first['calibration']
        changed.loc[held,'tb_status']=1-changed.loc[held,'tb_status'];changed.loc[held,'heart_rate']=200
        old=json.loads((Path(temp)/'outer_00/selection.json').read_text())
        cfg2=dict(cfg,output_dir=str(Path(temp)/'heldout_invariance'))
        v5.inner_search(changed,first,cfg2)
        new=json.loads((Path(cfg2['output_dir'])/'outer_00/selection.json').read_text())
        assert old==new
        # All fixed candidates' validation predictions must also be identical.
        for path in (Path(temp)/'outer_00/inner').glob('*.json'):
            a=json.loads(path.read_text());b=json.loads((Path(cfg2['output_dir'])/'outer_00/inner'/path.name).read_text())
            assert np.allclose(a['raw_probability'],b['raw_probability'],atol=1e-6)
        v5.finish_postprocessing(t,splits,cfg)
        xai=pd.read_csv(Path(temp)/'nested_grouped_xai.csv');assert xai.completeness_residual.abs().max()<1e-8
        assert (Path(temp)/'cascade_pooled_counts.csv').is_file()
        # Mutation of a saved model must be detected before deserialization.
        frozen=Path(temp)/'outer_00/models/beats_full_cb/bundle.joblib'
        original=frozen.read_bytes();frozen.write_bytes(original+b'corrupt')
        try:v5.load_bundle(frozen.parents[2],'beats_full_cb',cfg)
        except RuntimeError:pass
        else:raise AssertionError('Checkpoint tamper not detected')
        frozen.write_bytes(original)
        # No training reruns after postprocessing, and postprocessing resumes separately.
        v5.run_nested_cv(t,splits,cfg);v5.finish_postprocessing(t,splits,cfg)
        from src import coda_v5_plots as plots
        import matplotlib.pyplot as plt
        for fig in [plots.evaluation(metrics,pred,summary,pooled,paired,cfg),
                    plots.errors_calibration(pred,cfg),plots.diagnostics(pred,paired,cfg),plots.postprocessing(cfg)]:plt.close(fig)
        import shutil
        figures=root/'v5_validation/synthetic_figures';figures.mkdir(parents=True,exist_ok=True)
        for p in (Path(temp)/'figures').glob('*.png'):shutil.copyfile(p,figures/p.name)
        returned=v5.export_return(cfg,splits)
        import zipfile,hashlib
        with zipfile.ZipFile(returned) as z:
            assert z.testzip() is None
            for name,h in json.loads(z.read('run/artifact_checksums.json')).items():
                assert hashlib.sha256(z.read('run/'+name)).hexdigest()==h
        returned.unlink()
    assert all(cr.sha256(root/p)==h for p,h in source_hashes.items()),'Source changed during test; rerun final-source test.'
    result=dict(passed=True,synthetic_end_to_end=True,actual_beats_forward=True,actual_tabpfn_forward=True,
        all_model_families=True,all_44_candidates=True,heldout_invariance=True,resume=True,tamper_detection=True,
        bag_order_invariance=True,xai_completeness=True,return_export_crc_and_hashes=True,librosa_author_function_max_error=parity,
        no_real_TB_training=True,gpu_benchmark_performed=False,source_hashes=source_hashes,python=sys.version)
    dest=root/'v5_validation';dest.mkdir(exist_ok=True)
    (dest/'SYNTHETIC_TEST_RESULTS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
