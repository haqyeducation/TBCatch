"""Calibration-only cascade design, detailed XAI and matched baseline audit."""
from pathlib import Path
import json, copy, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from . import coda_research as cr
from . import coda_v3 as v3


def verify_reference(splits,cfg):
    folder=Path(cfg['package_root'])/'references/v3'
    hashes=json.loads((folder/'checksums.json').read_text())
    assert all(cr.sha256(folder/p)==h for p,h in hashes.items()),'Reference V3 checksum mismatch'
    old=json.loads((folder/'nested_splits.json').read_text())
    assert old==splits,'V4 splits must equal audited V3 splits.'
    target=Path(cfg['output_dir'])/'baseline_v3';target.mkdir(parents=True,exist_ok=True)
    import shutil
    for p in list(hashes)+['checksums.json']:shutil.copyfile(folder/p,target/p)
    return {'v3_splits_exact_match':True,'reference_files':len(hashes),
        'notice':'Frozen comparisons only; V3 outer outcomes are not used by inner selection.'}


def count_metrics(y,decisions):
    y=np.asarray(y);decisions=np.asarray(decisions,bool)
    tp=int(np.sum((y==1)&decisions));tn=int(np.sum((y==0)&~decisions))
    fn=int(np.sum((y==1)&~decisions));fp=int(np.sum((y==0)&decisions))
    return dict(n=len(y),tp=tp,fn=fn,tn=tn,fp=fp,sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),
        ppv=tp/(tp+fp) if tp+fp else np.nan,npv=tn/(tn+fn) if tn+fn else np.nan,
        referral_fraction=float(decisions.mean()))


def cascade_cutoff(y,p,route,target,fixed_mask=None,fixed_decisions=None):
    """Max specificity subject to whole-calibration sensitivity; no test use."""
    y=np.asarray(y);route=np.asarray(route,bool)
    candidates=np.r_[np.nextafter(p.max(),np.inf),np.unique(p[route])]
    feasible=[]
    for cut in candidates:
        decisions=route&(p>=cut)
        if fixed_mask is not None:decisions[fixed_mask]=fixed_decisions[fixed_mask]
        m=count_metrics(y,decisions)
        if m['sensitivity']>=target:feasible.append((m['specificity'],cut))
    if not feasible:raise RuntimeError('Cascade calibration target infeasible; do not silently lower the target.')
    return float(max(feasible)[1])


def staged_evaluation(bundles,table,fold,cfg):
    from . import coda_v4 as v4
    folder=Path(cfg['output_dir'])/f'outer_{fold["outer_fold"]:02d}'
    cal=table.loc[fold['calibration']];test=table.loc[fold['test']]
    first=bundles['audio_hr'];full=bundles['audio_clinical_hr']
    p1cal=v4.predict(first,cal);p2cal=v4.predict(full,cal)
    gate=v4.threshold(cal.tb_status,p1cal,'sensitivity90',cfg['gate_calibration_target'])
    routecal=(p1cal>=gate)|cal.heart_rate.isna().to_numpy()
    missingcal=cal.heart_rate.isna().to_numpy()
    fallbackcut=bundles['audio_clinical']['thresholds']['sensitivity90']
    fallbackcal=v4.predict(bundles['audio_clinical'],cal)>=fallbackcut
    cut=cascade_cutoff(cal.tb_status.to_numpy(),p2cal,routecal,cfg['target_sensitivity'],missingcal,fallbackcal)
    caldecisions=routecal&(p2cal>=cut);caldecisions[missingcal]=fallbackcal[missingcal]
    frozen={'gate_threshold':gate,'full_threshold':cut,'gate_calibration_target':cfg['gate_calibration_target'],
        'end_to_end_calibration_target':cfg['target_sensitivity'],'missing_hr_forces_questionnaire':True,
        'fallback_without_hr_threshold':fallbackcut,
        'calibration_metrics':count_metrics(cal.tb_status,caldecisions),
        'notice':'Calibration target not a test guarantee; low first-stage score is not a clinical rule-out.'}
    v4.atomic_json(folder/'CASCADE_FROZEN.json',frozen)
    rows=[];metrics=[]
    conditions={'original':test}
    for shift in cfg['hr_shifts_bpm']:
        t=test.copy();t['heart_rate']+=shift;t.loc[~t.heart_rate.between(30,220),'heart_rate']=np.nan
        conditions[f'hr_shift_{shift:+d}']=t
    for name,t in conditions.items():
        p1=v4.predict(first,t);route=(p1>=gate)|t.heart_rate.isna().to_numpy()
        # Fallback without HR: do not force the full-HR model's imputation path.
        p2=np.empty(len(t));missing=t.heart_rate.isna().to_numpy()
        p2[~missing]=v4.predict(full,t.iloc[np.flatnonzero(~missing)] ) if (~missing).any() else np.array([])
        if missing.any():p2[missing]=v4.predict(bundles['audio_clinical'],t.iloc[np.flatnonzero(missing)])
        # Fallback has its OWN calibration-selected cutoff. End-to-end target
        # above is designed on same fallback-aware calibration scores below.
        fallbackcut=bundles['audio_clinical']['thresholds']['sensitivity90']
        cuts=np.where(missing,fallbackcut,cut)
        decisions=route&(p2>=cuts)
        m=count_metrics(t.tb_status,decisions)
        metrics.append(dict(outer_fold=fold['outer_fold'],condition=name,questionnaire_fraction=float(route.mean()),
            stage1_missed_tb=int(np.sum((t.tb_status.to_numpy()==1)&~route)),**m))
        if name=='original':
            rows=[dict(outer_fold=fold['outer_fold'],participant=pid,y=int(y),stage1_probability=float(a),
                stage2_probability=float(b) if routed else np.nan,questionnaire_required=bool(routed),
                prediction=int(decision),hr_missing=bool(miss),gate_threshold=gate,full_threshold=float(c))
                for pid,y,a,b,routed,decision,miss,c in zip(t.index,t.tb_status,p1,p2,route,decisions,missing,cuts)]
    pd.DataFrame(rows).to_csv(folder/'cascade_predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(folder/'cascade_metrics.csv',index=False)


def _individual_shapley(bundle,table,pid,background,permutations,seed):
    from . import coda_v4 as v4
    fields={}
    for group,cols in bundle['groups'].items():
        if group=='audio':fields['audio']=cols
        else:
            for col in cols:fields[col]=[col]
    keys=list(fields);rng=np.random.default_rng(seed);frames=[];orders=[]
    for _ in range(permutations):
        order=list(rng.permutation(keys));orders.append(order)
        t=background.copy();frames.append(t.copy())
        for key in order:
            for col in fields[key]:t[col]=pd.Series([table.at[pid,col]]*len(t),index=t.index,dtype=table[col].dtype)
            frames.append(t.copy())
    values=v4.predict(bundle,pd.concat(frames)).reshape(permutations,len(keys)+1,len(background)).mean(axis=2)
    phi={k:0. for k in keys}
    for order,row in zip(orders,np.diff(values,axis=1)):
        for key,delta in zip(order,row):phi[key]+=float(delta)/permutations
    base=float(values[:,0].mean());prob=float(values[:,-1].mean())
    assert abs(prob-base-sum(phi.values()))<1e-8
    return phi,base,prob


def explain_individual(bundles,table,fold,cfg):
    folder=Path(cfg['output_dir'])/f'outer_{fold["outer_fold"]:02d}'
    definition=json.loads((folder/'xai_definition.json').read_text())
    b=bundles['audio_clinical_hr'];rows=[]
    for pid in definition['test_ids']:
        results=[]
        for repeat in range(2):
            bg=table.loc[definition['background_id_sets'][repeat][:cfg['individual_xai_background']]]
            results.append(_individual_shapley(b,table,pid,bg,cfg['individual_xai_permutations'],cfg['seed']+5000+repeat))
        a,base,prob=results[0];second=results[1][0]
        for key,value in a.items():rows.append(dict(outer_fold=fold['outer_fold'],participant=pid,feature=key,
            shapley=value,base_probability=base,probability=prob,repeat_abs_difference=abs(value-second[key]),
            method='Monte Carlo interventional Shapley; audio grouped; noncausal'))
    pd.DataFrame(rows).to_csv(folder/'individual_xai.csv',index=False)


def aggregate_extras(splits,cfg,predictions,metrics):
    out=Path(cfg['output_dir'])
    def concat(name):return pd.concat([pd.read_csv(out/f'outer_{f["outer_fold"]:02d}'/name) for f in splits['folds']],ignore_index=True)
    cascade=concat('cascade_predictions.csv');cm=concat('cascade_metrics.csv');individual=concat('individual_xai.csv')
    assert cascade.participant.is_unique and len(cascade)==splits['n_participants']
    cascade.to_csv(out/'nested_cascade_predictions.csv',index=False);cm.to_csv(out/'nested_cascade_metrics.csv',index=False)
    individual.to_csv(out/'nested_individual_xai.csv',index=False)
    pooled=count_metrics(cascade.y,cascade.prediction);pooled['questionnaire_fraction']=float(cascade.questionnaire_required.mean())
    pd.DataFrame([pooled]).to_csv(out/'cascade_pooled_counts.csv',index=False)
    summary=cm.groupby('condition')[['sensitivity','specificity','questionnaire_fraction','stage1_missed_tb']].agg(['mean','std'])
    summary.to_csv(out/'cascade_mean_std.csv')
    rows=[];rng=np.random.default_rng(cfg['seed'])
    p=predictions[predictions.policy=='youden']
    baseline=out/'baseline_v3/nested_oof_predictions.csv'
    comparisons=[('audio_clinical_hr','reference_mobile_lr'),('audio_clinical_hr','reference_mobile_cb'),
        ('audio_clinical_hr','reference_paper_cb'),('audio_hr','hr')]
    pairs=[]
    for a,b in comparisons:pairs.append((a+' minus '+b,p[p.combination==a],p[p.combination==b]))
    if baseline.exists():
        old=pd.read_csv(baseline,float_precision='round_trip');old=old[old.policy=='youden']
        for a in v3.COMBINATIONS:pairs.append((a+' V4 minus V3',p[p.combination==a],old[old.combination==a]))
    for name,a,b in pairs:
        a=a.sort_values('participant');b=b.sort_values('participant')
        assert a.participant.tolist()==b.participant.tolist() and np.array_equal(a.y,b.y)
        assert np.array_equal(a.outer_fold,b.outer_fold)
        y=a.y.to_numpy();pa=a.probability.to_numpy();pb=b.probability.to_numpy();boot=[]
        for _ in range(cfg['bootstrap_repeats']):
            ix=np.concatenate([rng.choice(np.flatnonzero(y==k),sum(y==k),replace=True) for k in [0,1]])
            boot.append(roc_auc_score(y[ix],pa[ix])-roc_auc_score(y[ix],pb[ix]))
        diffs=[roc_auc_score(a[a.outer_fold==f].y,a[a.outer_fold==f].probability)-
               roc_auc_score(b[b.outer_fold==f].y,b[b.outer_fold==f].probability) for f in sorted(a.outer_fold.unique())]
        rows.append(dict(comparison=name,pooled_auc_delta=roc_auc_score(y,pa)-roc_auc_score(y,pb),
            conditional_bootstrap_low=float(np.quantile(boot,.025)),conditional_bootstrap_high=float(np.quantile(boot,.975)),
            mean_fold_auc_delta=float(np.mean(diffs)),std_fold_auc_delta=float(np.std(diffs,ddof=1)),positive_folds=sum(x>0 for x in diffs)))
    paired=pd.DataFrame(rows);paired.to_csv(out/'matched_baseline_comparisons.csv',index=False)
    full=metrics[(metrics.combination=='audio_clinical_hr')&(metrics.policy=='youden')]
    v3.atomic_json(out/'RESEARCH_OUTCOME.json',{'full_mean_auc':float(full.roc_auc.mean()),
        'above_published_Sensors_rounded_081_descriptively':bool(full.roc_auc.mean()>.81),
        'independent_superiority_established':False,
        'notice':'Report all outcomes even if lower. Frozen V3 reuse and repeated development require independent confirmation. Paired bootstrap conditional, no multiplicity correction. Sensors-style reference is approximate SciPy, not exact replication.'})
    return cascade,cm,individual,paired
