"""Saved notebook visuals, using only the existing matplotlib stack."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve,confusion_matrix
from sklearn.calibration import calibration_curve
from scipy import signal
from . import coda_v5_features as features
from . import coda_v5_models as models

COLORS=['#216869','#49a078','#9cc5a1','#e9b44c','#9b5de5','#d1495b']
plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titleweight':'bold','axes.prop_cycle':plt.cycler(color=COLORS),'font.size':10})


def save(fig,name,cfg):
    path=Path(cfg['output_dir'])/'figures';path.mkdir(exist_ok=True)
    fig.tight_layout();fig.savefig(path/(name+'.png'),dpi=170,bbox_inches='tight')
    return fig


def show(fig):
    from IPython.display import display
    display(fig);plt.close(fig)


def eda(participants,manifest,cfg):
    t=features.encode_clinical(participants);fig,ax=plt.subplots(2,3,figsize=(16,9))
    counts=t.tb_status.value_counts().reindex([0,1],fill_value=0)
    ax[0,0].bar(['Non-TB','TB'],counts,color=COLORS[:2]);ax[0,0].set_title('Participant labels')
    for i,v in enumerate(counts):ax[0,0].text(i,v,str(v),ha='center',va='bottom')
    ax[0,1].hist(manifest.groupby('participant').size(),bins=25);ax[0,1].set_title('Coughs per participant (not a predictor)')
    missing=t[models.CLINICAL+['heart_rate']].isna().mean().sort_values()
    ax[0,2].barh(missing.index,100*missing);ax[0,2].set_title('Missing clinical fields (%)')
    for target,label in [(0,'Non-TB'),(1,'TB')]:
        subset=t[t.tb_status==target]
        ax[1,0].hist(subset.heart_rate.dropna(),bins=25,alpha=.5,label=label)
        ax[1,1].scatter(subset.age,subset.temperature,s=10,alpha=.3,label=label)
    ax[1,0].set_title('Heart rate (dataset measurement)');ax[1,0].legend()
    ax[1,1].set(xlabel='Age',ylabel='Temperature',title='Clinical distributions');ax[1,1].legend()
    corr=t[['age','height','weight','reported_cough_dur','heart_rate','temperature']].corr()
    im=ax[1,2].imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');ax[1,2].set_xticks(range(len(corr)),corr.columns,rotation=60,ha='right')
    ax[1,2].set_yticks(range(len(corr)),corr.columns);ax[1,2].set_title('Descriptive correlations');fig.colorbar(im,ax=ax[1,2],fraction=.05)
    return save(fig,'01_cohort_eda',cfg)


def audio_examples(manifest,cfg):
    fig,ax=plt.subplots(2,2,figsize=(13,7))
    for i,label in enumerate([0,1]):
        row=manifest[manifest.tb_status==label].sort_values('filename').iloc[0]
        x=features.read_physical(Path(cfg['data_root'])/row.path)
        ax[i,0].plot(np.arange(len(x))/16000,x,lw=.7);ax[i,0].set(title=f'Example {"TB" if label else "Non-TB"}: waveform',xlabel='Seconds',ylabel='PCM amplitude')
        f,t,z=signal.stft(x,16000,nperseg=512,noverlap=384)
        ax[i,1].pcolormesh(t,f,20*np.log10(np.maximum(abs(z),1e-6)),shading='auto',cmap='magma')
        ax[i,1].set(title='Spectrogram (illustration, not a diagnosis)',xlabel='Seconds',ylabel='Hz',ylim=(0,8000))
    return save(fig,'02_audio_examples',cfg)


def preprocessing(table,splits,cfg):
    fig,ax=plt.subplots(1,3,figsize=(16,4.5))
    labels=['Proper train','Calibration','Test'];folds=splits['folds'];bottom=np.zeros(len(folds))
    for key,label,color in zip(['proper_train','calibration','test'],labels,COLORS):
        n=np.array([len(f[key]) for f in folds]);ax[0].bar(np.arange(len(folds))+1,n,bottom=bottom,label=label,color=color);bottom+=n
    ax[0].set(title='Patient-disjoint outer folds',xlabel='Outer fold',ylabel='Participants');ax[0].legend(fontsize=8)
    for i,rep in enumerate(['paper','beats'],1):
        if '_bag_'+rep not in table:ax[i].text(.5,.5,'BEATs extraction on GPU PC',ha='center');continue
        x=np.concatenate(table['_bag_'+rep]);var=np.nanvar(x,axis=0)
        ax[i].hist(np.log10(np.maximum(var,1e-12)),bins=40,color=COLORS[i]);ax[i].set(title=f'{rep}: feature variance',xlabel='log10 variance',ylabel='Features')
    return save(fig,'03_preprocessing_audit',cfg)


def fold_overview(fold,cfg):
    path=Path(cfg['output_dir'])/f'outer_{fold["outer_fold"]:02d}'/'outer_test_metrics.csv'
    m=pd.read_csv(path);s=m[m.policy=='youden'].set_index('combination').reindex(['selected_full']+models.OUTPUTS)
    fig,ax=plt.subplots(figsize=(11,5));ax.barh(s.index,s.roc_auc,color=COLORS[0]);ax.invert_yaxis();ax.set(xlim=(0,1),xlabel='Test ROC-AUC',title=f'Outer {fold["outer_fold"]+1}: report only; never tune from these scores')
    return save(fig,f'outer_{fold["outer_fold"]:02d}',cfg)


def evaluation(metrics,predictions,summary,pooled,paired,cfg):
    s=summary[summary.policy=='youden'].set_index('combination').reindex(['selected_full']+models.OUTPUTS)
    fig,ax=plt.subplots(2,2,figsize=(17,11))
    y=np.arange(len(s));ax[0,0].barh(y,s.roc_auc_mean,xerr=s.roc_auc_std,color=COLORS[0],alpha=.9)
    ax[0,0].set_yticks(y,s.index);ax[0,0].invert_yaxis();ax[0,0].set(xlim=(0,1),title='Mean outer AUC ± sample SD (not CI)')
    ax[0,0].axvline(.81,ls='--',color=COLORS[5],label='Sensors reported ~0.81 (different cohort)');ax[0,0].legend(fontsize=8)
    selected=['clinical_full','paper_full_cb','beats_full_cb','selected_full','ensemble_full','beats_hr']
    for combo in selected:
        p=predictions[(predictions.combination==combo)&(predictions.policy=='youden')]
        fpr,tpr,_=roc_curve(p.y,p.probability);ax[0,1].plot(fpr,tpr,label=combo)
        precision,recall,_=precision_recall_curve(p.y,p.probability);ax[1,0].plot(recall,precision,label=combo)
    ax[0,1].plot([0,1],[0,1],'k--',alpha=.3);ax[0,1].set(title='Secondary pooled OOF ROC',xlabel='False positive rate',ylabel='Sensitivity');ax[0,1].legend(fontsize=8)
    ax[1,0].set(title='Secondary pooled precision–recall',xlabel='Recall',ylabel='Precision');ax[1,0].legend(fontsize=8)
    a=summary[summary.policy=='sensitivity90'].set_index('combination').reindex(selected)
    x=np.arange(len(a));ax[1,1].bar(x-.18,a.sensitivity_mean,.36,label='Sensitivity');ax[1,1].bar(x+.18,a.specificity_mean,.36,label='Specificity')
    ax[1,1].set_xticks(x,selected,rotation=35,ha='right');ax[1,1].set(ylim=(0,1.05),title='Fixed calibration-derived high-sensitivity policy');ax[1,1].legend()
    return save(fig,'04_evaluation',cfg)


def errors_calibration(predictions,cfg):
    fig,ax=plt.subplots(2,3,figsize=(15,9))
    for j,combo in enumerate(['paper_full_cb','beats_full_cb','selected_full']):
        for i,policy in enumerate(['youden','sensitivity90']):
            t=predictions[(predictions.combination==combo)&(predictions.policy==policy)]
            cm=confusion_matrix(t.y,t.prediction,labels=[0,1]);ax[i,j].imshow(cm,cmap='Blues')
            for (r,c),v in np.ndenumerate(cm):ax[i,j].text(c,r,str(v),ha='center',va='center',fontsize=16)
            ax[i,j].set_xticks([0,1],['Non-TB','TB']);ax[i,j].set_yticks([0,1],['Non-TB','TB'])
            ax[i,j].set(title=f'{combo}\n{policy}',xlabel='Prediction',ylabel='Reference')
    return save(fig,'05_confusion_matrices',cfg)


def diagnostics(predictions,paired,cfg):
    fig,ax=plt.subplots(1,2,figsize=(16,6))
    for combo in ['clinical_full','paper_full_cb','beats_full_cb','selected_full','ensemble_full']:
        p=predictions[(predictions.combination==combo)&(predictions.policy=='youden')]
        observed,predicted=calibration_curve(p.y,p.probability,n_bins=8,strategy='quantile')
        ax[0].plot(predicted,observed,'o-',label=combo)
    ax[0].plot([0,1],[0,1],'k--');ax[0].set(title='Pooled calibration (descriptive)',xlabel='Predicted probability',ylabel='Observed TB fraction');ax[0].legend(fontsize=8)
    for i,r in paired.iterrows():
        ax[1].plot([r.conditional_ci_low,r.conditional_ci_high],[i,i],color=COLORS[0]);ax[1].plot(r.pooled_auc_delta,i,'o',color=COLORS[1])
    ax[1].set_yticks(range(len(paired)),paired.comparison);ax[1].axvline(0,color='gray',ls='--');ax[1].set(title='Exploratory paired pooled AUC differences',xlabel='Δ AUC; conditional bootstrap 95% interval');ax[1].invert_yaxis()
    return save(fig,'06_calibration_paired',cfg)


def postprocessing(cfg):
    out=Path(cfg['output_dir']);xai=pd.read_csv(out/'nested_individual_xai.csv');robust=pd.read_csv(out/'nested_robustness.csv');cascade=pd.read_csv(out/'cascade_pooled_counts.csv').iloc[0]
    fig,ax=plt.subplots(1,3,figsize=(18,6))
    values=xai.groupby('feature').shapley.agg(lambda x:x.abs().mean()).sort_values()
    ax[0].barh(values.index,values,color=COLORS[0]);ax[0].set(title='Sampled individual |Shapley| (noncausal)',xlabel='Mean absolute contribution')
    m=robust[(robust.policy=='sensitivity90')&(robust.combination!='beats_hr')].groupby('condition')[['sensitivity','specificity']].mean()
    m.plot.bar(ax=ax[1],color=COLORS[:2]);ax[1].set(title='Selected full: HR shift simulation',ylim=(0,1));ax[1].tick_params(axis='x',rotation=40)
    frac=cascade.questionnaire_fraction
    ax[2].pie([frac,1-frac],labels=['Questionnaire','Skipped'],autopct='%1.1f%%',colors=COLORS[:2],wedgeprops={'width':.4})
    ax[2].set_title(f'Mobile cascade\nSens {cascade.sensitivity:.1%}, spec {cascade.specificity:.1%}\nTB stopped at stage 1: {int(cascade.stage1_missed_tb)}')
    return save(fig,'07_xai_robustness_cascade',cfg)
