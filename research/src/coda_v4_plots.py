"""Notebook figures and a standalone report for nested participant-level CV."""
from pathlib import Path
import base64,html,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve,confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from . import coda_research as cr
from .coda_v4 import LABELS,COMBINATIONS,CLINICAL,POLICIES

COLORS=['#167d9a','#d55e00','#299d75','#9163b6','#bd9425','#4778ba','#c65783','#6c757d','#8172b2','#64b5cd','#ccb974']


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.alpha':.17,'axes.facecolor':'white','figure.facecolor':'#fafcfe'})


def save(fig,name,cfg):
    folder=Path(cfg['output_dir'])/'figures';folder.mkdir(parents=True,exist_ok=True)
    fig.tight_layout(pad=1.8)
    for suffix in ['png','pdf']:fig.savefig(folder/(name+'.'+suffix),dpi=165,bbox_inches='tight')
    plt.close(fig);return folder/(name+'.png')


def show(paths):
    from IPython.display import Image,display
    for path in paths:display(Image(filename=str(path)))


def eda(participants,manifest,splits,table,cfg):
    style();paths=[];first=splits['folds'][0]
    proper=participants.set_index('participant').loc[first['proper_train']]
    fig,ax=plt.subplots(2,2,figsize=(13,8))
    counts=participants.tb_status.value_counts().reindex([0,1]);bars=ax[0,0].bar(['Non-TB','TB'],counts,color=COLORS[:2])
    ax[0,0].bar_label(bars);ax[0,0].set(title='Kohort lokal tersedia',ylabel='Peserta')
    country=pd.crosstab(participants.Country,participants.tb_status).reindex(columns=[0,1],fill_value=0)
    country.columns=['Non-TB','TB'];country.plot.bar(stacked=True,ax=ax[0,1],rot=0,color=COLORS[:2])
    ax[0,1].set(title='Negara dan label (deskriptif)',xlabel='Negara',ylabel='Peserta')
    for y,color in zip([0,1],COLORS):
        ax[1,0].hist(proper.loc[proper.tb_status==y,'heart_rate'].dropna(),bins=np.arange(30,181,5),
            alpha=.5,color=color,label=['Non-TB','TB'][y],density=True)
    ax[1,0].legend();ax[1,0].set(title='Denyut | proper train outer 1',xlabel='BPM',ylabel='Densitas')
    ax[1,1].hist(manifest.groupby('participant').size(),bins=25,color=COLORS[0])
    ax[1,1].set(title='Jumlah rekaman per peserta',xlabel='Batuk',ylabel='Peserta')
    paths.append(save(fig,'01_cohort_overview',cfg))
    fig,ax=plt.subplots(1,2,figsize=(14,5))
    x=np.arange(len(splits['folds']));bottom=np.zeros(len(x))
    for name,color in zip(['proper_train','calibration','test'],COLORS[:3]):
        heights=np.array([len(f[name]) for f in splits['folds']]);ax[0].bar(x,heights,bottom=bottom,color=color,label=name);bottom+=heights
    ax[0].set_xticks(x,x+1);ax[0].legend(fontsize=9);ax[0].set(title='Nested CV: kelompok pada setiap outer fold',xlabel='Outer fold',ylabel='Peserta')
    missing=proper[CLINICAL+['heart_rate']].isna().mean().mul(100).sort_values()
    ax[1].barh(missing.index,missing,color=COLORS[0]);ax[1].set(title='Missing | proper train outer 1',xlabel='% peserta')
    paths.append(save(fig,'02_nested_splits_and_missing',cfg))
    examples=[];m=manifest[manifest.participant.isin(first['proper_train'])]
    for y in [0,1]:examples+=m[m.tb_status==y].drop_duplicates('participant').head(2).to_dict('records')
    fig,ax=plt.subplots(len(examples),2,figsize=(13,2.4*len(examples)),squeeze=False)
    for i,row in enumerate(examples):
        wave=cr.read_audio(Path(cfg['data_root'])/row['path'],cfg)
        ax[i,0].plot(np.arange(len(wave))/16000,wave,color=COLORS[row['tb_status']],lw=.7)
        ax[i,0].set(title=f"{row['participant']} | {'TB' if row['tb_status'] else 'Non-TB'}",xlabel='Detik',ylabel='Amplitudo z-score')
        ax[i,1].specgram(wave,NFFT=512,Fs=16000,noverlap=352,cmap='magma')
        ax[i,1].set(title='Spektrogram ilustratif',xlabel='Detik',ylabel='Hz')
    paths.append(save(fig,'03_waveform_and_spectrogram',cfg))
    fig,ax=plt.subplots(1,2,figsize=(14,4.8))
    groups=['mfcc_audio_','rich_mean_','rich_std_'];dimensions=[sum(c.startswith(p) for c in table) for p in groups]
    b=ax[0].bar(['MFCC mean','Rich mean','Rich bag std'],dimensions,color=COLORS[:3]);ax[0].bar_label(b)
    ax[0].set(title='Dimensi sebelum PCA fold-local',ylabel='Fitur')
    cols=[c for c in table if c.startswith('rich_mean_')]
    proper_features=table.loc[first['proper_train'],cols]
    coords=PCA(n_components=2,random_state=cfg['seed']).fit_transform(StandardScaler().fit_transform(proper_features))
    labels=table.loc[first['proper_train'],'tb_status'].to_numpy()
    for y,color in zip([0,1],COLORS):ax[1].scatter(coords[labels==y,0],coords[labels==y,1],s=14,alpha=.4,color=color,label=['Non-TB','TB'][y])
    ax[1].legend();ax[1].set(title='PCA ilustratif | proper train outer 1 saja',xlabel='Komponen 1',ylabel='Komponen 2')
    paths.append(save(fig,'04_audio_features_illustration',cfg))
    return paths


def fold_overview(fold,cfg):
    folder=Path(cfg['output_dir'])/f'outer_{fold["outer_fold"]:02d}'
    metrics=pd.read_csv(folder/'outer_test_metrics.csv')
    style();fig,ax=plt.subplots(1,2,figsize=(14,4.8))
    auc=metrics[metrics.policy=='youden'].set_index('combination').loc[list(COMBINATIONS)]
    ax[0].barh([LABELS[c] for c in auc.index],auc.roc_auc,color=COLORS)
    ax[0].set(title=f'Outer {fold["outer_fold"]+1}: AUROC held-out',xlim=(0,1),xlabel='AUROC')
    for policy,color in zip(POLICIES,COLORS[:2]):
        s=metrics[(metrics.combination=='audio_clinical_hr')&(metrics.policy==policy)].iloc[0]
        ax[1].scatter(s.specificity,s.sensitivity,s=90,color=color,label=f'{policy}: FP {int(s.fp)}, FN {int(s.fn)}')
    ax[1].axhline(.9,color='gray',ls='--');ax[1].legend(fontsize=9)
    ax[1].set(title='Semua input: dua cutoff calibration',xlabel='Spesifisitas',ylabel='Sensitivitas',xlim=(0,1.03),ylim=(0,1.03))
    return [save(fig,f'outer_{fold["outer_fold"]+1:02d}_progress',cfg)]


def evaluation(metrics,predictions,summary,pooled,cfg):
    style();paths=[];combos=list(COMBINATIONS);labels=[LABELS[c] for c in combos]
    # Primary paper-comparable summary is outer-fold mean +/- sample SD.
    fig,ax=plt.subplots(1,2,figsize=(15,5))
    s=summary[summary.policy=='youden'].set_index('combination').loc[combos]
    ax[0].barh(labels,s.roc_auc_mean,color=COLORS)
    ax[0].errorbar(s.roc_auc_mean,np.arange(len(combos)),xerr=s.roc_auc_std,fmt='none',capsize=3,color='#243b50')
    ax[0].set(title='AUROC: mean +/- SD antar-outer-fold',xlabel='AUROC (SD bukan confidence interval)',xlim=(0,1.05))
    for combo,color in zip(combos,COLORS):
        sub=metrics[(metrics.combination==combo)&(metrics.policy=='youden')].sort_values('outer_fold')
        ax[1].plot(sub.outer_fold+1,sub.roc_auc,'o-',color=color,label=LABELS[combo],ms=3,lw=1)
    ax[1].set(title='Variasi AUROC antar-outer-fold',xlabel='Outer fold',ylabel='AUROC',ylim=(0,1.03));ax[1].legend(fontsize=8)
    paths.append(save(fig,'05_nested_auc_mean_sd',cfg))
    fig,ax=plt.subplots(1,2,figsize=(16,5))
    for j,(policy,a) in enumerate(zip(POLICIES,ax)):
        s=summary[summary.policy==policy].set_index('combination').loc[combos]
        y=np.arange(len(combos))
        a.barh(y-.17,s.sensitivity_mean,height=.34,color=COLORS[0],label='Sensitivitas')
        a.barh(y+.17,s.specificity_mean,height=.34,color=COLORS[1],label='Spesifisitas')
        a.set_yticks(y,labels);a.set_xlim(0,1.05);a.axvline(.9,color='gray',ls='--',lw=.8)
        a.set_title(f'{policy} | mean outer-fold');a.legend(fontsize=9)
    paths.append(save(fig,'06_two_cutoff_policies',cfg))
    fig,ax=plt.subplots(1,2,figsize=(14,6))
    for combo,color in zip(combos,COLORS):
        df=predictions[(predictions.combination==combo)&(predictions.policy=='youden')]
        fpr,tpr,_=roc_curve(df.y,df.probability);pr,rec,_=precision_recall_curve(df.y,df.probability)
        ax[0].plot(fpr,tpr,color=color,label=LABELS[combo]);ax[1].plot(rec,pr,color=color,label=LABELS[combo])
    ax[0].plot([0,1],[0,1],'--',color='gray');ax[0].set(title='Pooled OOF ROC (sekunder)',xlabel='False positive rate',ylabel='Sensitivitas')
    ax[1].set(title='Pooled OOF precision-recall (sekunder)',xlabel='Recall',ylabel='Precision / PPV')
    for a in ax:a.legend(fontsize=8);a.set_xlim(0,1);a.set_ylim(0,1.03)
    paths.append(save(fig,'07_pooled_oof_roc_pr',cfg))
    for policy in POLICIES:
        fig,ax=plt.subplots(3,4,figsize=(18,12))
        for combo,a in zip(combos,ax.flat):
            df=predictions[(predictions.combination==combo)&(predictions.policy==policy)]
            cm=confusion_matrix(df.y,df.prediction,labels=[0,1]);scaled=cm/cm.sum(axis=1,keepdims=True)
            a.imshow(scaled,vmin=0,vmax=1,cmap='Blues')
            for (i,j),count in np.ndenumerate(cm):a.text(j,i,f'{["TN","FP","FN","TP"][2*i+j]}\n{count}',ha='center',va='center',fontsize=13,color='white' if scaled[i,j]>.55 else '#21394f')
            a.set(title=LABELS[combo],xlabel='Prediksi',ylabel='Aktual',xticks=[0,1],yticks=[0,1],xticklabels=['Non-TB','TB'],yticklabels=['Non-TB','TB']);a.grid(False)
        ax.flat[-1].axis('off');fig.suptitle(f'OOF confusion matrices | {policy} | cutoff berbeda sesuai calibration tiap fold',fontsize=13)
        paths.append(save(fig,'08_confusion_'+policy,cfg))
    fig,ax=plt.subplots(1,2,figsize=(14,5))
    for combo,color in zip(['audio','audio_hr','clinical_hr','audio_clinical_hr'],COLORS):
        df=predictions[(predictions.combination==combo)&(predictions.policy=='youden')]
        actual,prob=calibration_curve(df.y,df.probability,n_bins=6,strategy='quantile')
        ax[0].plot(prob,actual,'o-',color=color,label=LABELS[combo])
    ax[0].plot([0,1],[0,1],'--',color='gray');ax[0].legend(fontsize=8)
    ax[0].set(title='Reliability pooled OOF (sekunder)',xlabel='Rata-rata probabilitas',ylabel='Proporsi TB aktual',xlim=(0,1),ylim=(0,1))
    data=[metrics[(metrics.combination=='audio_clinical_hr')&(metrics.policy==p)].threshold.to_numpy() for p in POLICIES]
    ax[1].boxplot(data,tick_labels=POLICIES,showmeans=True)
    ax[1].set(title='Variasi cutoff model lengkap antar-fold',ylabel='Cutoff dari calibration')
    paths.append(save(fig,'09_calibration_and_cutoff_variability',cfg))
    fig,ax=plt.subplots(1,2,figsize=(16,5))
    for policy,a in zip(POLICIES,ax):
        s=pooled[pooled.policy==policy].set_index('combination').loc[combos]
        a.barh(labels,s.referral_fraction*100,color=COLORS);a.set(xlim=(0,100),xlabel='% peserta ditandai positif',title=f'Beban rujukan | {policy}')
    paths.append(save(fig,'10_referral_fraction',cfg))
    return paths


def diagnostics(metrics,paired,xai,robust,cfg):
    style();paths=[]
    fig,ax=plt.subplots(figsize=(12,5))
    x=paired.pooled_auc_delta.to_numpy();low=paired.conditional_bootstrap_low.to_numpy();high=paired.conditional_bootstrap_high.to_numpy()
    ax.errorbar(x,np.arange(len(x)),xerr=np.stack([np.maximum(0,x-low),np.maximum(0,high-x)]),fmt='o',capsize=5,color=COLORS[0])
    ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(len(x)),paired.comparison)
    ax.set(title='Nilai tambahan input | bootstrap conditional eksploratif',xlabel='Delta pooled OOF AUROC dan 95% CI')
    paths.append(save(fig,'11_paired_input_comparisons',cfg))
    fig,ax=plt.subplots(1,2,figsize=(15,5))
    means=xai.groupby('combination')[['audio','clinical','hr']].agg(lambda s:s.abs().mean()).reindex(list(COMBINATIONS))
    means.index=[LABELS[c] for c in means.index]
    means.plot.barh(stacked=True,ax=ax[0],color=COLORS[:3]);ax[0].set(title='XAI sesuai model setiap outer fold',xlabel='Mean |group Shapley| pada sampel tetap')
    cols=[g+'_background_abs_delta' for g in ['audio','clinical','hr']]
    delta=xai.groupby('combination')[cols].mean().reindex(list(COMBINATIONS));delta.index=means.index
    delta.columns=['Audio','Kuesioner','Denyut'];delta.plot.barh(ax=ax[1],color=COLORS[:3])
    ax[1].set(title='Sensitivitas XAI terhadap sampel background',xlabel='Mean |perbedaan Shapley| dua background')
    paths.append(save(fig,'12_xai_and_background_stability',cfg))
    fig,ax=plt.subplots(1,2,figsize=(15,5))
    for combo,color in zip(['audio_hr','clinical_hr','audio_clinical_hr'],COLORS):
        sub=robust[(robust.combination==combo)&(robust.policy=='sensitivity90')&robust.condition.str.startswith('hr_shift')].copy()
        sub['shift']=sub.condition.str.replace('hr_shift_','',regex=False).astype(int)
        mean=sub.groupby('shift').specificity.mean();ax[0].plot(mean.index,mean,'o-',label=LABELS[combo],color=color)
    ax[0].legend(fontsize=8);ax[0].set(title='Simulasi HR | mean outer-fold spesifisitas',xlabel='Perubahan BPM (bukan error kamera terukur)',ylabel='Spesifisitas',ylim=(0,1))
    sub=robust[(robust.policy=='youden')&(robust.condition=='audio_permuted')].groupby('combination').mean_abs_probability_change.mean()
    ax[1].barh([LABELS[c] for c in sub.index],sub,color=COLORS[:len(sub)])
    ax[1].set(title='Ketergantungan pada audio yang dipertukarkan',xlabel='Mean |perubahan skor|')
    paths.append(save(fig,'13_hr_and_audio_robustness',cfg))
    # The selection frequency is itself based solely on inner development.
    selections=[]
    for path in sorted(Path(cfg['output_dir']).glob('outer_*/selection.json')):
        selected=json.loads(path.read_text())['winners'];search=pd.read_csv(path.parent/'inner_search.csv').set_index('id')
        selections.extend({'combination':combo,'representation':search.loc[cid,'representation'],
            'family':search.loc[cid,'family'],'fusion':search.loc[cid,'fusion']} for combo,cid in selected.items())
    df=pd.DataFrame(selections);counts=pd.crosstab(df.combination,df.representation).reindex(list(COMBINATIONS));counts.index=[LABELS[c] for c in counts.index]
    fig,ax=plt.subplots(figsize=(12,5));counts.plot.barh(stacked=True,ax=ax,color=COLORS)
    ax.set(title='Representasi terpilih pada setiap outer fold',xlabel='Jumlah outer fold')
    paths.append(save(fig,'14_selected_representations',cfg))
    return paths


def report(cfg,summary,pooled,paired):
    out=Path(cfg['output_dir'])
    body=['<h1>CODA TB V4 — Nested participant CV</h1>',
        '<p>Internal exploratory evaluation. Same local cohort was examined in V1/V2/V3. No official hidden test or camera HR validation. Models/cutoffs selected without outer-test labels within each fold.</p>',
        '<h2>Primary: mean and sample SD over outer folds</h2>',summary.to_html(index=False,float_format=lambda x:f'{x:.4f}'),
        '<h2>Secondary: pooled OOF decisions using fold-specific frozen cutoffs</h2>',pooled.to_html(index=False,float_format=lambda x:f'{x:.4f}'),
        '<h2>Paired exploratory comparisons</h2>',paired.to_html(index=False,float_format=lambda x:f'{x:.4f}'),
        '<p>Bootstrap is conditional on trained fold models; no multiplicity correction. Count intervals are descriptive and do not account for all dependencies from overlapping training sets. SD is not a 95% CI. No deployment model is chosen from the best outer-test fold.</p>']
    for path in sorted((out/'figures').glob('*.png')):
        body+=['<h2>'+html.escape(path.stem.replace('_',' '))+'</h2>',
            '<img style="width:100%;max-width:1600px" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">']
    text='<!doctype html><meta charset="utf-8"><title>CODA TB V4</title><style>body{font:16px system-ui;margin:35px;color:#19334c}table{font-size:11px;border-collapse:collapse}td,th{padding:5px;border:1px solid #ddd}</style>'+''.join(body)
    target=out/'REPORT.html';target.write_text(text,encoding='utf-8');return target


def extra_figures(cfg,summary,extra_results):
    cascade,cm,individual,matched=extra_results;paths=[];style()
    fig,ax=plt.subplots(2,2,figsize=(16,11))
    s=summary[summary.policy=='youden'].set_index('combination')
    names=['audio_clinical_hr','reference_mobile_lr','reference_mobile_cb','reference_paper_lr','reference_paper_cb']
    ax[0,0].barh([LABELS[k] for k in names],[s.loc[k,'roc_auc_mean'] for k in names],
        xerr=[s.loc[k,'roc_auc_std'] for k in names],color=COLORS[:5],capsize=3)
    ax[0,0].axvline(.81,ls=':',color='grey',label='Sensors published 0.81 (different cohort)')
    ax[0,0].set(title='Model lengkap vs referensi lokal | mean ± SD',xlabel='AUC; referensi SciPy bukan replikasi persis',xlim=(0,1))
    ax[0,0].legend(fontsize=8)
    d=cm[cm.condition=='original'].sort_values('outer_fold')
    ax[0,1].plot(d.outer_fold+1,d.sensitivity,'o-',label='Sensitivitas akhir')
    ax[0,1].plot(d.outer_fold+1,d.specificity,'s-',label='Spesifisitas akhir')
    ax[0,1].plot(d.outer_fold+1,d.questionnaire_fraction,'^-',label='Proporsi kuesioner')
    ax[0,1].axhline(.9,color='grey',ls=':');ax[0,1].set(title='Alur bertahap | seluruh peserta test',xlabel='Outer fold',ylim=(0,1.05));ax[0,1].legend(fontsize=8)
    imp=individual.groupby('feature').shapley.agg(lambda x:x.abs().mean()).sort_values()
    ax[1,0].barh(imp.index,imp,color=COLORS[0]);ax[1,0].set(title='Individual XAI model lengkap | sampel tetap',xlabel='Mean |Monte Carlo Shapley|; bukan kausal')
    select=[]
    for path in sorted(Path(cfg['output_dir']).glob('outer_*/selection.json')):
        winners=json.loads(path.read_text())['winners'];search=pd.read_csv(path.parent/'inner_search.csv').set_index('id')
        select.extend({'combination':c,'engine':search.loc[cid,'engine']} for c,cid in winners.items())
    counts=pd.crosstab(pd.DataFrame(select).combination,pd.DataFrame(select).engine)
    counts.index=[LABELS[c] for c in counts.index];counts.plot.barh(stacked=True,ax=ax[1,1])
    ax[1,1].set(title='Engine terpilih melalui inner CV',xlabel='Jumlah outer fold')
    paths.append(save(fig,'15_v4_references_cascade_individual_xai',cfg))
    fig,ax=plt.subplots(figsize=(13,6))
    ax.errorbar(matched.pooled_auc_delta,np.arange(len(matched)),
        xerr=np.stack([np.maximum(0,matched.pooled_auc_delta-matched.conditional_bootstrap_low),
                       np.maximum(0,matched.conditional_bootstrap_high-matched.pooled_auc_delta)]),fmt='o',capsize=4)
    ax.axvline(0,color='grey',ls='--');ax.set_yticks(np.arange(len(matched)),matched.comparison)
    ax.set(title='Perbandingan berpasangan V4 | eksploratif',xlabel='Delta pooled OOF AUC; conditional 95% bootstrap interval')
    paths.append(save(fig,'16_v4_matched_comparisons',cfg))
    return paths


_core_report=report
def report(cfg,summary,pooled,paired):
    target=_core_report(cfg,summary,pooled,paired);text=target.read_text(encoding='utf-8')
    extra=['<h2>V4: staged screening and matched baselines</h2>',
        '<p>Mobile selection prioritizes inner AUC; reference searches prioritize UAR. The 261-dimensional SciPy features approximate Sensors feature families, not exact LibROSA replication. Published 0.81 is a descriptive benchmark on a different cohort. No independent superiority claim.</p>']
    for name in ['RESEARCH_OUTCOME.json','cascade_pooled_counts.csv','cascade_mean_std.csv','matched_baseline_comparisons.csv']:
        p=Path(cfg['output_dir'])/name
        if p.exists():
            extra.append('<h3>'+html.escape(name)+'</h3>')
            extra.append('<pre>'+html.escape(p.read_text())+'</pre>' if p.suffix=='.json' else pd.read_csv(p).to_html(index=False))
    extra.append('<p>Gate target is not guaranteed sensitivity. Missing HR routes to a separately trained model without HR. No camera-reference, prospective-user, or external cohort validation has been performed. Individual Shapley is sampled, noncausal, with audio grouped. Training on real TB cohort occurs only in full mode.</p>')
    target.write_text(text+''.join(extra),encoding='utf-8');return target
