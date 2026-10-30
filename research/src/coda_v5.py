"""V5: fixed-budget patient-disjoint CV, full clinical ablations, resumable audit."""
from pathlib import Path
import itertools, json, math, os, re, time, zipfile
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from . import coda_research as cr
from . import coda_v4 as protocol
from . import coda_v5_models as models
from . import coda_v5_features as features

atomic_json=protocol.atomic_json
atomic_model=protocol.atomic_model
digest=protocol.digest
POLICIES=['youden','sensitivity90']
audit_cohort=protocol.audit_cohort
nested_splits=protocol.nested_splits
threshold=protocol.threshold
calculate_metrics=protocol.calculate_metrics


def configure(package,run_name,mode='full',project_root=None):
    import importlib.metadata as md
    import torch
    if mode not in ['full','preflight'] or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,70}',run_name):raise ValueError('Invalid mode/run name')
    package=Path(package).resolve();cfg=cr.load_config(package)
    cfg.update(json.loads((package/'config_v5.json').read_text()))
    root=Path(project_root).resolve() if project_root else package.parent
    cfg.update(data_root=str(root/'dataset/dataset'),
               manifest=str(root/'dataset-audit/local-full/solicited_training_manifest_with_review_flags.csv'),
               package_root=str(package),output_dir=str(package/'runs'/run_name),mode=mode)
    for k in ['data_root','manifest']:
        if not Path(cfg[k]).exists():raise FileNotFoundError(f'{k}: {cfg[k]}')
    features.verify_assets(package)
    versions={p:md.version(p) for p in ['torch','scikit-learn','numpy','scipy','pandas','catboost','librosa','tabpfn','torchaudio','joblib']}
    if versions['tabpfn']!='2.0.9' or versions['librosa']!='0.10.2.post1':
        raise RuntimeError('Use pinned V5 local libraries: tabpfn 2.0.9, librosa 0.10.2.post1. See setup notebook.')
    if mode=='full' and cfg['require_cuda_for_full'] and not torch.cuda.is_available():raise RuntimeError('MODE=full requires CUDA on training PC; use preflight here.')
    if torch.cuda.is_available():
        features.gpu_guard(cfg,'configure')
        total=torch.cuda.get_device_properties(0).total_memory
        fraction=min(.9,(cfg['vram_limit_gb']-cfg['vram_headroom_gb'])*1e9/total)
        torch.cuda.set_per_process_memory_fraction(fraction,0)
    cfg['source_hashes']={p.relative_to(package).as_posix():cr.sha256(p) for p in
        [package/'config.json',package/'config_v5.json',package/'assets_v5.json',
         package/'src/coda_research.py',*sorted((package/'src').glob('coda_v[345]*.py')),
         *sorted((package/'src/vendor_beats').glob('*.py'))]}
    cfg['input_hashes']={'manifest':cr.sha256(cfg['manifest']),**{
        p.name:cr.sha256(p) for p in (Path(cfg['data_root'])/'meta_data/Clinical').glob('*.csv')}}
    reference=package/'references/v5_v4'
    if reference.exists():
        for name,h in json.loads((reference/'checksums.json').read_text()).items():
            if cr.sha256(reference/name)!=h:raise RuntimeError('V4 reference checksum mismatch: '+name)
        cfg['input_hashes']['v4_reference_checksums']=cr.sha256(reference/'checksums.json')
    cfg['versions']=versions
    cfg['protocol_fingerprint']=digest({k:v for k,v in cfg.items() if k not in ['package_root','output_dir','data_root','manifest']})
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if (out/'resolved_config.json').exists():
        old=json.loads((out/'resolved_config.json').read_text())
        if old['protocol_fingerprint']!=cfg['protocol_fingerprint']:raise RuntimeError('Run differs in source/data/library/config. Use a new RUN_NAME; old checkpoints are preserved.')
    atomic_json(out/'resolved_config.json',cfg)
    if not (out/'environment.json').exists():cr.environment_report(cfg)
    atomic_json(out/'candidate_grid.json',models.candidate_grid(cfg))
    cr.seed_everything(cfg['seed']);torch.set_num_threads(min(cfg['cpu_threads'],os.cpu_count() or 1))
    return cfg


def load_checkpoint(path,signature):
    path=Path(path)
    if not path.exists():return None
    check=path.with_suffix(path.suffix+'.sha256.json')
    if not check.exists():raise RuntimeError('Incomplete checkpoint metadata: '+str(path))
    meta=json.loads(check.read_text())
    if meta['signature']!=signature or cr.sha256(path)!=meta['sha256']:raise RuntimeError('Checkpoint changed: '+str(path))
    return joblib.load(path)


def save_checkpoint(path,value,signature):
    atomic_model(path,value)
    atomic_json(Path(path).with_suffix(Path(path).suffix+'.sha256.json'),{'signature':signature,'sha256':cr.sha256(path)})


def inner_search(table,fold,cfg):
    folder=Path(cfg['output_dir'])/f'outer_{fold["outer_fold"]:02d}';folder.mkdir(parents=True,exist_ok=True)
    sig=digest({'protocol':cfg['protocol_fingerprint'],'fold':fold});grid=models.candidate_grid(cfg)
    records=[];scores=[];time_rows=[]
    for i,c in enumerate(grid):
        candidate_rows=[];stats=[]
        for j,sp in enumerate(fold['inner_folds']):
            path=folder/'inner'/f'{c["id"]}_{j}.json'
            if path.exists():
                r=json.loads(path.read_text())
                if r['signature']!=sig or r['candidate']!=c or r['train_ids']!=sp['train']:raise RuntimeError('Inner checkpoint mismatch')
            else:
                start=time.perf_counter()
                if c['family']=='blend':
                    parts=[]
                    for cid in c['components']:
                        source=json.loads((folder/'inner'/f'{cid}_{j}.json').read_text())
                        if source['signature']!=sig or source['validation_ids']!=sp['validation']:raise RuntimeError('Blend source mismatch')
                        parts.append(np.array(source['raw_probability']))
                    p=sum(w*v for w,v in zip(c['weights'],parts))
                else:
                    model=models.fit_model(table,sp['train'],c,cfg,cfg['seed']+1000*fold['outer_fold']+j)
                    try:p=models.predict_raw(model,table.loc[sp['validation']],cfg)
                    finally:models.release(model)
                y=table.loc[sp['validation'],'tb_status'].to_numpy(int)
                cut=threshold(y,p,'youden');m=calculate_metrics(y,p,cut)
                r=dict(signature=sig,candidate=c,train_ids=sp['train'],validation_ids=sp['validation'],
                    y=y.tolist(),raw_probability=p.tolist(),inner_fold=j,auc=m['roc_auc'],uar=m['uar'],
                    ranking_cutoff=cut,seconds=time.perf_counter()-start)
                atomic_json(path,r)
            if r['validation_ids']!=sp['validation'] or r['y']!=table.loc[sp['validation'],'tb_status'].astype(int).tolist():raise RuntimeError('OOF identity mismatch')
            if not np.isfinite(r['raw_probability']).all():raise ValueError('Bad cached probability')
            stats.append(r)
            candidate_rows.extend(dict(candidate=c['id'],participant=pid,y=int(y),raw_probability=float(p),inner_fold=j)
                for pid,y,p in zip(r['validation_ids'],r['y'],r['raw_probability']))
            time_rows.append(dict(candidate=c['id'],family=c['family'],inner_fold=j,seconds=r['seconds']))
            print(f'Outer {fold["outer_fold"]+1}/{cfg["outer_folds"]} | {i+1}/{len(grid)} {c["id"]} | inner {j+1}/{len(fold["inner_folds"])}',flush=True)
        assert len(candidate_rows)==len(fold['proper_train']) and len({r['participant'] for r in candidate_rows})==len(candidate_rows)
        scores.append(dict(id=c['id'],output=c['output'],family=c['family'],
            mean_inner_uar=float(np.mean([r['uar'] for r in stats])),mean_inner_auc=float(np.mean([r['auc'] for r in stats]))))
        records.extend(candidate_rows)
        pd.DataFrame(scores).to_csv(folder/'inner_search.csv',index=False)
    score=pd.DataFrame(scores);oof=pd.DataFrame(records)
    winners={k:str(score[score.output==k].sort_values(['mean_inner_uar','mean_inner_auc','id'],ascending=[False,False,True]).iloc[0].id) for k in models.OUTPUTS}
    # A single full model for post-hoc XAI, chosen before test labels are read.
    full=score[score.output.isin(['beats_full_lr','beats_full_cb','beats_full_tabpfn','ensemble_full'])]
    primary=full.sort_values(['mean_inner_uar','mean_inner_auc','id'],ascending=[False,False,True]).iloc[0].output
    atomic_json(folder/'selection.json',dict(winners=winners,primary_full_output=primary,rule=cfg['selection_rule']))
    oof.to_csv(folder/'proper_oof.csv',index=False);pd.DataFrame(time_rows).to_csv(folder/'timing.csv',index=False)
    return oof,winners


def predict(bundle,table,cfg):
    return np.clip(bundle['calibrator'].predict(models.predict_raw(bundle['model'],table,cfg)),0,1)


def finalize_fold(table,fold,oof,winners,cfg):
    folder=Path(cfg['output_dir'])/f'outer_{fold["outer_fold"]:02d}';grid={c['id']:c for c in models.candidate_grid(cfg)}
    frozen={}
    for output,cid in winners.items():
        sig=digest({'protocol':cfg['protocol_fingerprint'],'fold':fold,'candidate':grid[cid]})
        path=folder/'models'/output/'bundle.joblib'
        bundle=load_checkpoint(path,sig)
        if bundle is None:
            selected=oof[oof.candidate==cid].sort_values('participant')
            assert set(selected.participant)==set(fold['proper_train'])
            iso=IsotonicRegression(y_min=0,y_max=1,increasing=True,out_of_bounds='clip').fit(selected.raw_probability,selected.y)
            model=models.fit_model(table,fold['proper_train'],grid[cid],cfg,cfg['seed']+1000*fold['outer_fold']+99)
            bundle=dict(model=model,calibrator=iso,candidate=grid[cid],groups=model['groups'],
                        proper_ids=fold['proper_train'],calibration_ids=fold['calibration'],protocol_fingerprint=cfg['protocol_fingerprint'])
            try:
                cal=table.loc[fold['calibration']];raw=models.predict_raw(model,cal,cfg);p=iso.predict(raw)
                bundle['thresholds']={policy:threshold(cal.tb_status,p,policy,cfg['target_sensitivity']) for policy in POLICIES}
                frame=pd.DataFrame(dict(participant=cal.index,y=cal.tb_status,raw_probability=raw,probability=p))
            finally:models.release(model)
            save_checkpoint(path,bundle,sig)
            frame.to_csv(path.parent/'calibration_predictions.csv',index=False)
            atomic_json(path.parent/'metadata.json',{k:v for k,v in bundle.items() if k not in ['model','calibrator']})
        frozen[output]={'candidate':grid[cid],'thresholds':bundle['thresholds'],'signature':sig,'sha256':cr.sha256(path)}
        models.release(bundle['model'])
    atomic_json(folder/'MODELS_AND_THRESHOLDS_FROZEN.json',frozen)


def load_bundle(folder,output,cfg):
    record=json.loads((Path(folder)/'MODELS_AND_THRESHOLDS_FROZEN.json').read_text())[output]
    path=Path(folder)/'models'/output/'bundle.joblib'
    if cr.sha256(path)!=record['sha256']:raise RuntimeError('Frozen model changed')
    return load_checkpoint(path,record['signature'])


def evaluate_fold(table,fold,cfg):
    folder=Path(cfg['output_dir'])/f'outer_{fold["outer_fold"]:02d}';test=table.loc[fold['test']]
    predictions=[];metrics=[]
    for output in models.OUTPUTS:
        b=load_bundle(folder,output,cfg)
        try:
            raw=models.predict_raw(b['model'],test,cfg);p=b['calibrator'].predict(raw)
            for policy,cut in b['thresholds'].items():
                metrics.append(dict(outer_fold=fold['outer_fold'],combination=output,policy=policy,
                    raw_roc_auc=roc_auc_score(test.tb_status,raw),**calculate_metrics(test.tb_status,p,cut)))
                predictions.extend(dict(outer_fold=fold['outer_fold'],combination=output,policy=policy,
                    participant=pid,y=int(y),probability=float(v),raw_probability=float(r),threshold=cut,prediction=int(v>=cut))
                    for pid,y,v,r in zip(test.index,test.tb_status,p,raw))
        finally:models.release(b['model'])
    primary=json.loads((folder/'selection.json').read_text())['primary_full_output']
    # Report the pre-test nested selection as the primary V5 procedure too.
    predictions.extend([dict(r,combination='selected_full') for r in list(predictions) if r['combination']==primary])
    metrics.extend([dict(r,combination='selected_full') for r in list(metrics) if r['combination']==primary])
    pd.DataFrame(predictions).to_csv(folder/'outer_test_predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(folder/'outer_test_metrics.csv',index=False)


def run_nested_cv(table,splits,cfg,on_fold=None):
    for fold in splits['folds']:
        folder=Path(cfg['output_dir'])/f'outer_{fold["outer_fold"]:02d}'
        complete=folder/'TRAIN_EVAL_COMPLETE.json'
        if complete.exists():
            record=json.loads(complete.read_text())
            if record['protocol']!=cfg['protocol_fingerprint']:raise RuntimeError('Fold fingerprint mismatch')
            for p,h in record['checksums'].items():
                if cr.sha256(folder/p)!=h:raise RuntimeError('Completed fold artifact changed: '+p)
            print('Reuse completed outer',fold['outer_fold']+1,flush=True)
        else:
            oof,winners=inner_search(table,fold,cfg)
            finalize_fold(table,fold,oof,winners,cfg)
            evaluate_fold(table,fold,cfg)
            checks={p.relative_to(folder).as_posix():cr.sha256(p) for p in folder.rglob('*') if p.is_file() and not p.name.endswith('.part')}
            atomic_json(complete,dict(protocol=cfg['protocol_fingerprint'],checksums=checks))
        if on_fold:on_fold(fold)


def paired_comparison(a,b,repeats,seed):
    a=a.sort_values('participant');b=b.sort_values('participant')
    assert a.participant.tolist()==b.participant.tolist() and np.array_equal(a.y,b.y)
    y=a.y.to_numpy();pa=a.probability.to_numpy();pb=b.probability.to_numpy();rng=np.random.default_rng(seed);boot=[]
    for _ in range(repeats):
        ix=np.concatenate([rng.choice(np.flatnonzero(y==k),int((y==k).sum()),replace=True) for k in [0,1]])
        boot.append(roc_auc_score(y[ix],pa[ix])-roc_auc_score(y[ix],pb[ix]))
    return dict(pooled_auc_delta=roc_auc_score(y,pa)-roc_auc_score(y,pb),
                conditional_ci_low=np.quantile(boot,.025),conditional_ci_high=np.quantile(boot,.975))


def aggregate_results(splits,cfg):
    out=Path(cfg['output_dir'])
    def concat(name):return pd.concat([pd.read_csv(out/f'outer_{f["outer_fold"]:02d}'/name,float_precision='round_trip') for f in splits['folds']],ignore_index=True)
    m=concat('outer_test_metrics.csv');p=concat('outer_test_predictions.csv')
    for _,g in p.groupby(['combination','policy']):assert len(g)==splits['n_participants'] and g.participant.is_unique
    cols=['roc_auc','raw_roc_auc','pr_auc','uar','sensitivity','specificity','ppv','npv','brier','referral_fraction']
    summary=m.groupby(['combination','policy'])[cols].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(x).rstrip('_') for x in summary.columns]
    pooled=[]
    for (combo,policy),g in p.groupby(['combination','policy']):
        y=g.y.to_numpy();pred=g.prediction.to_numpy();tp=int(((y==1)&(pred==1)).sum());fn=int(((y==1)&(pred==0)).sum())
        tn=int(((y==0)&(pred==0)).sum());fp=int(((y==0)&(pred==1)).sum())
        pooled.append(dict(combination=combo,policy=policy,n=len(g),roc_auc=roc_auc_score(y,g.probability),
            tp=tp,fn=fn,tn=tn,fp=fp,sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),
            ppv=tp/(tp+fp) if tp+fp else np.nan,referral_fraction=float(pred.mean())))
    pairs=[('selected_full','paper_full_cb'),('selected_full','paper_full_lr'),('selected_full','clinical_full'),
           ('beats_full_lr','paper_full_lr'),('beats_full_cb','paper_full_cb'),
           ('beats_full_lr','paper_pooled_lr'),('beats_full_cb','paper_pooled_cb'),
           ('beats_full_tabpfn','paper_full_tabpfn'),('beats_full_cb','clinical_full'),
           ('beats_full_tabpfn','clinical_full'),('ensemble_full','paper_full_cb'),
           ('ensemble_full','beats_full_cb'),('beats_mobile','beats_hr')]
    rows=[];unique=p[p.policy=='youden']
    for a,b in pairs:
        row=paired_comparison(unique[unique.combination==a],unique[unique.combination==b],cfg['bootstrap_repeats'],cfg['seed'])
        da=m[(m.combination==a)&(m.policy=='youden')].set_index('outer_fold').roc_auc
        db=m[(m.combination==b)&(m.policy=='youden')].set_index('outer_fold').roc_auc
        rows.append(dict(comparison=a+' minus '+b,mean_fold_auc_delta=float((da-db).mean()),positive_folds=int((da>db).sum()),**row))
    frames={'nested_fold_metrics.csv':m,'nested_oof_predictions.csv':p,'nested_mean_std.csv':summary,
            'pooled_oof_metrics.csv':pd.DataFrame(pooled),'paired_comparisons.csv':pd.DataFrame(rows)}
    for name,frame in frames.items():frame.to_csv(out/name,index=False)
    atomic_json(out/'interpretation.json',dict(primary='Mean +/- sample SD over 10 outer folds; SD is not CI',
        bootstrap='Conditional on fitted predictions; exploratory, no retraining or multiplicity correction',
        paper_comparison='Same 16 variables and LibROSA descriptor formulas; local cohort/splits/search differ; not exact published reproduction',
        prior_exposure='Same cohort used repeatedly in V1-V4; independent validation required',
        camera_hr_validated=False,deployment_model_selected=False))
    return m,p,summary,pd.DataFrame(pooled),pd.DataFrame(rows)


def _substitute(background,patient,columns):
    result=background.copy()
    for col in columns:
        value=patient[col]
        if isinstance(value,np.ndarray):result[col]=pd.Series([value]*len(result),index=result.index,dtype=object)
        else:result[col]=float(value)
    return result


def shapley_groups(bundle,table,pid,bg,groups,cfg,seed=0,permutations=None):
    keys=list(groups);n=len(keys);patient=table.loc[pid];frames=[]
    if permutations is None:
        paths=list(itertools.product([0,1],repeat=n))
    else:
        rng=np.random.default_rng(seed);orders=[rng.permutation(n).tolist() for _ in range(permutations)]
        needed={(0,)*n,(1,)*n}
        for order in orders:
            bits=[0]*n
            for j in order:bits[j]=1;needed.add(tuple(bits))
        paths=sorted(needed)
    for bits in paths:
        cols=sum([groups[k] for k,bit in zip(keys,bits) if bit],[])
        frames.append(_substitute(bg,patient,cols))
    raw=models.predict_raw(bundle['model'],pd.concat(frames),cfg)
    if not np.isfinite(raw).all():raise ValueError('Invalid XAI scores')
    # Canonical endpoint uses a single prediction of the actual participant.
    # Check raw agreement before calibration: isotonic jumps can amplify roundoff.
    values=dict(zip(paths,bundle['calibrator'].predict(raw).reshape(len(paths),len(bg)).mean(axis=1)))
    batch_full=values[(1,)*n]
    single_raw=float(models.predict_raw(bundle['model'],table.loc[[pid]],cfg)[0])
    full_index=paths.index((1,)*n)
    raw_delta=float(np.max(abs(raw.reshape(len(paths),len(bg))[full_index]-single_raw)))
    if raw_delta>1e-5:raise RuntimeError('XAI raw batch discrepancy >1e-5; keep training checkpoints and inspect predictor.')
    single=float(bundle['calibrator'].predict([single_raw])[0]);values[(1,)*n]=single;full=single
    phi={k:0. for k in keys}
    if permutations is None:
        for j,k in enumerate(keys):
            for bits,value in values.items():
                if bits[j]:continue
                size=sum(bits);other=list(bits);other[j]=1
                phi[k]+=math.factorial(size)*math.factorial(n-size-1)/math.factorial(n)*(values[tuple(other)]-value)
    else:
        for order in orders:
            bits=[0]*n;previous=values[tuple(bits)]
            for j in order:
                bits[j]=1;now=values[tuple(bits)];phi[keys[j]]+=(now-previous)/permutations;previous=now
    base=values[(0,)*n];residual=full-base-sum(phi.values())
    if abs(residual)>1e-8:raise RuntimeError('XAI completeness mismatch')
    return dict(base_probability=base,probability=full,single_probability=single,
                batch_abs_delta=abs(single-batch_full),raw_batch_abs_delta=raw_delta,completeness_residual=residual,**phi)


def postprocess_fold(table,fold,cfg):
    folder=Path(cfg['output_dir'])/f'outer_{fold["outer_fold"]:02d}'
    marker=folder/'POSTPROCESS_COMPLETE.json'
    if marker.exists():
        r=json.loads(marker.read_text())
        if r['protocol']!=cfg['protocol_fingerprint'] or any(cr.sha256(folder/p)!=h for p,h in r['checksums'].items()):raise RuntimeError('Postprocess artifact changed')
        return
    selection=json.loads((folder/'selection.json').read_text());outputs=[selection['primary_full_output'],'beats_hr']
    rng=np.random.default_rng(cfg['seed']+3000+fold['outer_fold'])
    ids=sorted(rng.choice(fold['test'],min(cfg['xai_per_outer_fold'],len(fold['test'])),replace=False).tolist())
    bgs=[rng.choice(fold['proper_train'],min(cfg['xai_background'],len(fold['proper_train'])),replace=False).tolist() for _ in range(2)]
    xai=[];individual=[];robust=[]
    for output in outputs:
        b=load_bundle(folder,output,cfg)
        try:
            for pid in ids:
                a=shapley_groups(b,table,pid,table.loc[bgs[0]],b['groups'],cfg)
                c=shapley_groups(b,table,pid,table.loc[bgs[1]],b['groups'],cfg)
                xai.append(dict(outer_fold=fold['outer_fold'],combination=output,participant=pid,**a,
                    **{k+'_background_abs_delta':abs(a[k]-c[k]) for k in b['groups']}))
            if output==selection['primary_full_output']:
                groups={'audio':b['groups']['audio'],**{c:[c] for c in models.CLINICAL+['heart_rate']}}
                for pid in ids[:2]:
                    # Hold permutations fixed to isolate background sensitivity.
                    seed=cfg['seed']+5000+fold['outer_fold']
                    a=shapley_groups(b,table,pid,table.loc[bgs[0][:2]],groups,cfg,seed,cfg['individual_xai_permutations'])
                    c=shapley_groups(b,table,pid,table.loc[bgs[1][:2]],groups,cfg,seed,cfg['individual_xai_permutations'])
                    for key in groups:individual.append(dict(outer_fold=fold['outer_fold'],combination=output,participant=pid,feature=key,
                        shapley=a[key],background_abs_difference=abs(a[key]-c[key]),completeness_residual=a['completeness_residual']))
            test=table.loc[fold['test']];conditions={'original':test}
            for shift in cfg['hr_shifts_bpm']:
                changed=test.copy();changed['heart_rate']+=shift;changed.loc[~changed.heart_rate.between(30,220),'heart_rate']=np.nan
                conditions[f'hr_shift_{shift:+d}']=changed
            for name,t in conditions.items():
                p=predict(b,t,cfg)
                for policy,cut in b['thresholds'].items():robust.append(dict(outer_fold=fold['outer_fold'],combination=output,condition=name,policy=policy,**calculate_metrics(test.tb_status,p,cut)))
        finally:models.release(b['model'])
    pd.DataFrame(xai).to_csv(folder/'grouped_xai.csv',index=False)
    pd.DataFrame(individual).to_csv(folder/'individual_xai.csv',index=False)
    pd.DataFrame(robust).to_csv(folder/'robustness.csv',index=False)
    # Mobile cascade uses the trained mobile questionnaire model, never the full
    # research model with height/weight/temp silently missing.
    first=load_bundle(folder,'beats_hr',cfg);second=load_bundle(folder,'beats_mobile',cfg)
    try:
        cal=table.loc[fold['calibration']];test=table.loc[fold['test']]
        p1=predict(first,cal,cfg);p2=predict(second,cal,cfg)
        gate=threshold(cal.tb_status,p1,'sensitivity90',cfg['gate_calibration_target'])
        routed=(p1>=gate)|cal.heart_rate.isna().to_numpy()
        eligible=[np.nextafter(float(p2.max()),np.inf),*np.unique(p2)[::-1],0.]
        ys=cal.tb_status.to_numpy();valid=[]
        for cut in eligible:
            decision=routed&(p2>=cut);sens=decision[ys==1].mean()
            if sens>=cfg['target_sensitivity']:valid.append((float((~decision[ys==0]).mean()),float(cut)))
        if not valid:raise RuntimeError('No feasible calibrated cascade cutoff')
        cut=max(valid)[1]
        atomic_json(folder/'CASCADE_FROZEN.json',dict(gate=gate,stage2_cutoff=cut,model='beats_mobile',
            target=cfg['target_sensitivity'],notice='Low score is not a clinical rule-out; camera HR unvalidated'))
        a=predict(first,test,cfg);b=predict(second,test,cfg);route=(a>=gate)|test.heart_rate.isna().to_numpy();decision=route&(b>=cut)
        frame=pd.DataFrame(dict(participant=test.index,y=test.tb_status,questionnaire_required=route,prediction=decision.astype(int),
            stage1_probability=a,stage2_probability=b,outer_fold=fold['outer_fold']))
        frame.to_csv(folder/'cascade_predictions.csv',index=False)
    finally:models.release(first['model']);models.release(second['model'])
    atomic_json(folder/'xai_definition.json',dict(test_ids=ids,background_ids=bgs,outputs=outputs,
        method='Interventional Shapley; grouped exact, individual Monte Carlo; not causal or clinically validated',
        individual_permutation_seed_fixed_across_backgrounds=True))
    names=['grouped_xai.csv','individual_xai.csv','robustness.csv','cascade_predictions.csv','CASCADE_FROZEN.json','xai_definition.json']
    atomic_json(marker,dict(protocol=cfg['protocol_fingerprint'],checksums={p:cr.sha256(folder/p) for p in names}))


def finish_postprocessing(table,splits,cfg):
    out=Path(cfg['output_dir'])
    for fold in splits['folds']:
        print('Postprocess outer',fold['outer_fold']+1,flush=True);postprocess_fold(table,fold,cfg)
    for name in ['grouped_xai','individual_xai','robustness','cascade_predictions']:
        frames=[pd.read_csv(out/f'outer_{f["outer_fold"]:02d}'/(name+'.csv')) for f in splits['folds']]
        pd.concat(frames,ignore_index=True).to_csv(out/('nested_'+name+'.csv'),index=False)
    c=pd.read_csv(out/'nested_cascade_predictions.csv');y=c.y.to_numpy();p=c.prediction.to_numpy()
    tp=int(((y==1)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum());tn=int(((y==0)&(p==0)).sum());fp=int(((y==0)&(p==1)).sum())
    pd.DataFrame([dict(tp=tp,fn=fn,tn=tn,fp=fp,sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),
        questionnaire_fraction=float(c.questionnaire_required.mean()),stage1_missed_tb=int(((y==1)&~c.questionnaire_required).sum()))]).to_csv(out/'cascade_pooled_counts.csv',index=False)


def export_return(cfg,splits):
    out=Path(cfg['output_dir']);package=Path(cfg['package_root'])
    for f in splits['folds']:
        folder=out/f'outer_{f["outer_fold"]:02d}'
        for name in ['TRAIN_EVAL_COMPLETE.json','POSTPROCESS_COMPLETE.json']:
            if not (folder/name).exists():raise RuntimeError('Incomplete fold: '+str(folder))
    atomic_json(out/'RUN_COMPLETE.json',dict(protocol=cfg['protocol_fingerprint'],completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        internal_exploratory=True,camera_hr_validated=False,deployment_model_selected=False))
    checks={p.relative_to(out).as_posix():cr.sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_checksums.json' and not p.name.endswith('.part')}
    atomic_json(out/'artifact_checksums.json',checks)
    target=out.with_name(out.name+'_RETURN.zip');tmp=target.with_suffix('.part')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for rel in checks:z.write(out/rel,'run/'+rel)
        z.write(out/'artifact_checksums.json','run/artifact_checksums.json')
        for rel in cfg['source_hashes']:z.write(package/rel,'research_package/'+rel)
        for rel in ['CODA_TB_V5_Research.ipynb','V5_README.md']:
            if (package/rel).exists():z.write(package/rel,'research_package/'+rel)
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None:raise RuntimeError('RETURN CRC failed')
    tmp.replace(target);return target
