"""Regression tests: injected batch drift, true mismatch, CPU TabPFN, no-retrain resume."""
from pathlib import Path
import copy,json,shutil,tempfile,zipfile,sys
from unittest.mock import patch
import numpy as np
import pandas as pd
import joblib,torch
from sklearn.isotonic import IsotonicRegression
from src import coda_research as cr,coda_v5 as v5,coda_v5_models as models
from src import v5_xai_resume as fix


def main():
    package=Path(__file__).resolve().parent
    cfg=cr.load_config(package);cfg.update(json.loads((package/'config_v5.json').read_text()))
    cfg.update(package_root=str(package),tabpfn_device='cpu',tabpfn_estimators=1,cpu_threads=2,
        outer_folds=2,inner_folds=2,require_cuda_for_full=False,xai_per_outer_fold=1,
        xai_background=2,individual_xai_permutations=2,protocol_fingerprint='SYNTHETIC_XAI_FIX')
    torch.set_num_threads(2);rng=np.random.default_rng(42);n=40
    t=pd.DataFrame(index=[f'SYNTH_{i:03d}' for i in range(n)])
    t['tb_status']=np.arange(n)%2;t['Country']='SYNTH'
    for col in models.CLINICAL:t[col]=rng.normal(size=n)+t.tb_status
    t['heart_rate']=80+10*t.tb_status+rng.normal(size=n)
    t['_bag_beats']=pd.Series([rng.normal(size=(3,8)) for _ in range(n)],index=t.index,dtype=object)
    grid=models.candidate_grid(cfg)
    c=next(c for c in grid if c['output']=='beats_full_lr')
    model=models.fit_model(t,t.index[:24],c,cfg,42)
    def bundle(model):return dict(model=model,groups=model['groups'],calibrator=IsotonicRegression(out_of_bounds='clip').fit([0,1],[0,1]))
    bg=t.iloc[:3];pid=t.index[-1];real=fix._RAW;original_shapley=fix._SHAPLEY
    expected=original_shapley(bundle(model),t,pid,bg,model['groups'],cfg)
    calls=[]
    def biased(m,frame,settings=None):
        calls.append(len(frame))
        return np.clip(real(m,frame,settings)+(2e-4 if len(frame)>1 else 0),0,1)
    try:
        fix._RAW=biased;models.predict_raw=biased
        try:original_shapley(bundle(model),t,pid,bg,model['groups'],cfg)
        except RuntimeError as e:assert 'batch discrepancy' in str(e)
        else:raise AssertionError('Original failure was not reproduced')
        b=bundle(model);actual=fix.stable_shapley(b,t,pid,bg,b['groups'],cfg)
        assert actual['rowwise_fallback_used'] and actual['original_batch_raw_delta']>1e-5
        for key in ['audio','clinical','hr','probability','base_probability']:
            assert abs(actual[key]-expected[key])<1e-10,(key,actual[key],expected[key])
        again=fix.stable_shapley(b,t,pid,bg,b['groups'],cfg)
        assert again['pointwise_cache_misses']==0
        assert models.predict_raw is biased
        wrong={k:v for k,v in b['groups'].items() if k!='hr'}
        try:fix.stable_shapley(b,t,pid,bg,wrong,cfg)
        except RuntimeError as e:assert 'cover exactly' in str(e)
        else:raise AssertionError('Missing model inputs not rejected')
        def broken(*args,**kwargs):raise RuntimeError('Deliberate test failure')
        fix._SHAPLEY=broken
        try:fix.stable_shapley(bundle(model),t,pid,bg,model['groups'],cfg)
        except RuntimeError:pass
        else:raise AssertionError('Expected exception')
        assert models.predict_raw is biased
        fix._SHAPLEY=original_shapley
        counter=[0]
        def unstable(m,frame,settings=None):
            counter[0]+=1
            return np.clip(real(m,frame,settings)+counter[0]*.001,0,1)
        fix._RAW=unstable;models.predict_raw=unstable
        try:fix.stable_shapley(bundle(model),t,pid,bg,model['groups'],cfg)
        except RuntimeError as e:assert 'not repeatable' in str(e)
        else:raise AssertionError('Unstable pointwise inference was hidden')
    finally:fix._RAW=real;models.predict_raw=real;fix._SHAPLEY=original_shapley
    # Real local TabPFN, no fake neural predictor or real TB labels.
    c=next(c for c in grid if c['output']=='beats_full_tabpfn')
    neural=models.fit_model(t,t.index[:24],c,cfg,42);b=bundle(neural)
    b['_v5_xai_runtime']={'cache':{},'checked':set(),'pointwise':True,'probes':0}
    r=fix.stable_shapley(b,t,pid,t.iloc[:1],b['groups'],cfg)
    assert r['pointwise_full_raw_delta']<=1e-10 and abs(r['completeness_residual'])<=1e-8
    models.release(neural)
    with tempfile.TemporaryDirectory(prefix='v5_xai_resume_') as temp:
        p=Path(temp)/'research_package';run='experiment_v5_synthetic';out=p/'runs'/run;out.mkdir(parents=True)
        config=copy.deepcopy(cfg);config.update(package_root=str(p),output_dir=str(out),versions={})
        config['source_hashes']={}
        for name in ['src/coda_v5.py','src/coda_v5_models.py','config_v5.json']:
            target=p/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(package/name,target)
            config['source_hashes'][name]=cr.sha256(target)
        v5.atomic_json(out/'resolved_config.json',config)
        t.reset_index(names='participant').to_csv(out/'cohort.csv',index=False)
        v5.atomic_model(out/'participant_features.joblib',t)
        splits=v5.nested_splits(t.reset_index(names='participant'),config)
        for f in splits['folds']:
            folder=out/f'outer_{f["outer_fold"]:02d}';folder.mkdir()
            frozen={};winners={};rows=[]
            for output in ['beats_full_lr','beats_hr','beats_mobile']:
                c=next(c for c in grid if c['output']==output and c['family']=='lr')
                m=models.fit_model(t,f['proper_train'],c,config,42);b=bundle(m)
                cal=t.loc[f['calibration']];prob=real(m,cal,config)
                b.update(thresholds={policy:v5.threshold(cal.tb_status,prob,policy,.9) for policy in v5.POLICIES},candidate=c)
                sig='SYNTHETIC_'+output;path=folder/'models'/output/'bundle.joblib';v5.save_checkpoint(path,b,sig)
                frozen[output]=dict(signature=sig,sha256=cr.sha256(path));winners[output]=c['id']
                test=t.loc[f['test']];prob=real(m,test,config)
                rows.extend(dict(combination=output,policy='youden',participant=i,y=int(y),raw_probability=float(x),
                    probability=float(x),threshold=b['thresholds']['youden'],prediction=int(x>=b['thresholds']['youden'])) for i,y,x in zip(test.index,test.tb_status,prob))
            v5.atomic_json(folder/'MODELS_AND_THRESHOLDS_FROZEN.json',frozen)
            v5.atomic_json(folder/'selection.json',dict(winners=winners,primary_full_output='beats_full_lr'))
            pd.DataFrame(rows).to_csv(folder/'outer_test_predictions.csv',index=False)
            checks={x.relative_to(folder).as_posix():cr.sha256(x) for x in folder.rglob('*') if x.is_file()}
            v5.atomic_json(folder/'TRAIN_EVAL_COMPLETE.json',dict(protocol=config['protocol_fingerprint'],checksums=checks))
        reports=[]
        for path in out.glob('outer_*/outer_test_predictions.csv'):
            for combo,g in pd.read_csv(path).groupby('combination'):
                reports.append(dict(combination=combo,policy='youden',**v5.calculate_metrics(g.y,g.probability,float(g.threshold.iloc[0]))))
        summary=pd.DataFrame(reports).groupby(['combination','policy'])[['roc_auc','sensitivity','specificity']].agg(['mean','std']).reset_index()
        summary.columns=['_'.join(c).rstrip('_') for c in summary.columns]
        summary.to_csv(out/'nested_mean_std.csv',index=False)
        loaded,table,saved,protected=fix.load_run(p,run)
        with patch.object(models,'fit_model',side_effect=AssertionError('Classifier refit attempted')):
            v5.finish_postprocessing(table,saved,loaded)
            fix.verify_unchanged(loaded,protected)
            summary=fix.summarize(loaded)
            assert summary['completeness_max_abs']<1e-8
            v5.finish_postprocessing(table,saved,loaded)
        returned=v5.export_return(loaded,saved)
        with zipfile.ZipFile(returned) as z:
            assert z.testzip() is None
            assert any(n.startswith('run/runtime_sources/v5_xai_resume_') for n in z.namelist())
        returned.unlink()
        # Execute every cell of the actual resume notebook in a fresh kernel.
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
        shutil.copyfile(package/'src/v5_xai_resume.py',p/'src/v5_xai_resume.py')
        nb=nbformat.read(package/'CODA_TB_V5_Resume_XAI.ipynb',as_version=4)
        for cell in nb.cells:
            if cell.cell_type=='code':
                cell.source=cell.source.replace('PACKAGE_DIR = ""','PACKAGE_DIR = '+repr(str(p)))
                cell.source=cell.source.replace('RUN_NAME = "experiment_v5_001"','RUN_NAME = '+repr(run))
        kernelroot=Path(temp)/'kernels';spec=kernelroot/'v5-xai-test';spec.mkdir(parents=True)
        (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'V5 XAI test','language':'python'}))
        km=KernelManager(kernel_name='v5-xai-test',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernelroot)]))
        client=NotebookClient(nb,km=km,timeout=180,resources={'metadata':{'path':str(package)}})
        try:client.execute()
        finally:
            if km.has_kernel:km.shutdown_kernel(now=True)
        assert not [o for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='error']
        evidence=package/'v5_fix_validation';evidence.mkdir(exist_ok=True)
        nbformat.write(nb,evidence/'SYNTHETIC_RESUME_NOTEBOOK_EXECUTED.ipynb')
        audit=json.loads((out/'V5_XAI_RUNTIME_FIX.json').read_text())
        assert audit['completed_postprocess_folds_before_fix']==[] and audit['applications']==2
        # A changed original source must never be accepted as the old protocol.
        (p/'config_v5.json').write_text('{}')
        try:fix.apply(loaded,saved)
        except RuntimeError as e:assert 'source changed' in str(e)
        else:raise AssertionError('Original-source change accepted')
    v5.shapley_groups=original_shapley
    result=dict(passed=True,original_error_reproduced=True,batch_drift_fallback=True,
        rowwise_matches_reference=True,cache_reuse=True,missing_inputs_rejected=True,
        nonrepeatable_predictor_rejected=True,predictor_restored_after_exception=True,
        actual_tabpfn_cpu_test=True,postprocessing_resume_without_classifier_training=True,
        protected_training_files_unchanged=True,return_contains_runtime_source=True,all_resume_notebook_cells_executed=True,
        source_change_rejected=True,gpu_reproduction=False,source_sha256=cr.sha256(fix.__file__))
    dest=package/'v5_fix_validation';dest.mkdir(exist_ok=True)
    v5.atomic_json(dest/'TEST_RESULTS.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
