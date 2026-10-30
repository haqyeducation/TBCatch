"""Read-only audit of local CODA data; outputs go to dataset-audit/local-full."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import collections, csv, hashlib, json, wave, zipfile, time
import numpy as np
import pandas as pd
import openpyxl

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT/'data/dataset/dataset'
OUT = PROJECT/'data/dataset-audit/local-full'
OUT.mkdir(parents=True, exist_ok=True)
start = time.time()
all_files = sorted(p for p in ROOT.rglob('*') if p.is_file())
print('Inventory', len(all_files), flush=True)
clinical = pd.read_csv(ROOT/'meta_data/Clinical/CODA_TB_Clinical_Meta_Info.csv')
extra = pd.read_csv(ROOT/'meta_data/Clinical/CODA_TB_additional_variables_train.csv')
sol = pd.read_csv(ROOT/'meta_data/Cough Metadata/CODA_TB_Solicited_Meta_Info.csv')
long = pd.read_csv(ROOT/'meta_data/Cough Metadata/CODA_TB_Longitudnal_Meta_Info.csv')
tables = {'clinical': clinical, 'additional': extra, 'solicited': sol, 'longitudinal': long}
summary = {'root': str(ROOT.resolve()), 'files':len(all_files), 'bytes':sum(p.stat().st_size for p in all_files),
           'extensions':dict(collections.Counter(p.suffix.lower() for p in all_files)), 'tables':{}}
for name, df in tables.items():
    summary['tables'][name] = {'rows':len(df),'columns':list(df.columns),'missing':df.isna().sum().to_dict(),
        'duplicate_rows':int(df.duplicated().sum()),'participants':int(df.participant.nunique()) if 'participant' in df else None}
    if 'participant' in df:
        summary['tables'][name]['unknown_participants'] = sorted(set(df.participant.dropna())-set(clinical.participant))
    if name in ['clinical','additional']:
        summary['tables'][name]['duplicate_participant_ids']=int(df.participant.duplicated().sum())
        summary['tables'][name]['values']={c:df[c].value_counts(dropna=False).to_dict() for c in df if df[c].nunique()<=25}
        summary['tables'][name]['numeric_description']=df.select_dtypes(include='number').describe().to_dict()
dictionary=[]
wb=openpyxl.load_workbook(ROOT/'meta_data/CODA TB Challenge data dictionary_update.xlsx',read_only=True,data_only=True)
for sheet in wb:
    dictionary.append({'sheet':sheet.title,'rows':list(sheet.values)})
(OUT/'data_dictionary.json').write_text(json.dumps(dictionary,default=str,indent=2),encoding='utf-8')

def audit_audio(path):
    row={'path':path.relative_to(ROOT).as_posix(),'filename':path.name,'subset':'solicited' if 'solicited_data' in path.parts else 'longitudinal','bytes':path.stat().st_size}
    try:
        data=path.read_bytes();row['sha256']=hashlib.sha256(data).hexdigest()
        with wave.open(str(path),'rb') as w:
            row.update(channels=w.getnchannels(),sample_rate=w.getframerate(),sample_width=w.getsampwidth(),frames=w.getnframes(),compression=w.getcomptype())
            raw=w.readframes(w.getnframes())
        row['duration']=row['frames']/row['sample_rate'];row['decoded_bytes']=len(raw)
        row['expected_pcm_bytes']=row['frames']*row['channels']*row['sample_width']
        row['truncated']=row['decoded_bytes']!=row['expected_pcm_bytes']
        if row['sample_width']==2:
            x=np.frombuffer(raw,dtype='<i2').astype(np.float64)/32768
            row['pcm_sha256']=hashlib.sha256(raw).hexdigest()
            row.update(peak=float(np.max(np.abs(x))) if len(x) else 0,rms=float(np.sqrt(np.mean(x*x))) if len(x) else 0,
                zero_fraction=float(np.mean(x==0)) if len(x) else 1,clipping_fraction=float(np.mean((x<=-1)|(x>=32767/32768))) if len(x) else 0,
                dc_offset=float(np.mean(x)) if len(x) else 0)
        row['error']=''
    except Exception as e:row['error']=str(e)
    return row
audio=[p for p in all_files if p.suffix.lower()=='.wav']
rows=[]
with ThreadPoolExecutor(max_workers=8) as pool:
    for i,row in enumerate(pool.map(audit_audio,audio)):
        rows.append(row)
        if (i+1)%10000==0:print('Decoded',i+1,flush=True)
aud=pd.DataFrame(rows);aud.to_csv(OUT/'audio_inventory.csv',index=False)
summary['audio']={}
for subset,df in aud.groupby('subset'):
    summary['audio'][subset]={'files':len(df),'bytes':int(df.bytes.sum()),'duration_hours':float(df.duration.sum()/3600),
        'errors':int((df.error!='').sum()),'truncated':int(df.truncated.fillna(False).sum()),
        'sample_rates':df.sample_rate.value_counts().to_dict(),'channels':df.channels.value_counts().to_dict(),
        'sample_width':df.sample_width.value_counts().to_dict(),'duration_stats':df.duration.describe().to_dict(),
        'silent':int((df.rms==0).sum()),'rms_below_0_001':int((df.rms<0.001).sum()),
        'any_clipping':int((df.clipping_fraction>0).sum()),'clipping_above_1pct':int((df.clipping_fraction>0.01).sum())}
    meta=tables[subset];available=set(df.filename);expected=set(meta.filename)
    summary['audio'][subset].update(metadata_unique_filenames=len(expected),missing_files=len(expected-available),unreferenced_files=len(available-expected),
        duplicate_disk_filenames=int(df.filename.duplicated().sum()),duplicate_metadata_filenames=int(meta.filename.duplicated().sum()))
    meta[~meta.filename.isin(available)].to_csv(OUT/f'{subset}_missing_audio.csv',index=False)
    df[~df.filename.isin(expected)].to_csv(OUT/f'{subset}_unreferenced_audio.csv',index=False)
    mapping=meta.merge(df,on='filename',how='inner')
    summary['audio'][subset]['available_participants']=int(mapping.participant.nunique())
    summary['audio'][subset]['missing_participants']=sorted(set(meta.participant)-set(mapping.participant))
    counts=mapping.groupby('participant').size().rename('available_coughs').reset_index().merge(clinical,on='participant',how='left').merge(extra,on='participant',how='left')
    counts.to_csv(OUT/f'{subset}_participant_manifest.csv',index=False)
    summary['audio'][subset]['participants_by_label']=counts.tb_status.value_counts(dropna=False).to_dict()
    summary['audio'][subset]['coughs_per_participant']=counts.available_coughs.describe().to_dict()
    summary['audio'][subset]['country_label_counts']=counts.groupby(['Country','tb_status']).size().to_dict()
    if subset=='solicited':
        mapping.merge(clinical,on='participant',how='left').to_csv(OUT/'solicited_training_manifest_unfiltered.csv',index=False)
    if 'sound_prediction_score' in meta:
        summary['audio'][subset]['sound_prediction_score']=meta.sound_prediction_score.describe().to_dict()
        summary['audio'][subset]['scores_below_0_8']=int((meta.sound_prediction_score<0.8).sum())
    nums=pd.to_numeric(meta.filename.str.extract(r'^(\d+)')[0],errors='coerce')
    dt=pd.to_datetime(nums,unit='ms',utc=True,errors='coerce')
    summary['audio'][subset]['filename_inferred_dates']={'min':str(dt.min()),'max':str(dt.max()),'years':dt.dt.year.value_counts().to_dict()}

mapping=pd.concat([tables[s][['participant','filename']].assign(subset=s) for s in ['solicited','longitudinal']],ignore_index=True).drop_duplicates()
hashmap=aud.merge(mapping,on=['filename','subset'],how='left')
duplicates=[]
for hashval,g in hashmap.groupby('pcm_sha256'):
    if len(g)>1:
        d=g.copy();d['duplicate_group']=hashval;duplicates.append(d)
dup=pd.concat(duplicates,ignore_index=True) if duplicates else pd.DataFrame()
dup.to_csv(OUT/'duplicate_pcm_records.csv',index=False)
summary['duplicates']={'groups':len(duplicates),'records':len(dup),'cross_subset_groups':sum(g.subset.nunique()>1 for g in duplicates),
    'cross_participant_groups':sum(g.participant.nunique()>1 for g in duplicates)}
merged=clinical.merge(extra,on='participant',how='outer',indicator=True)
summary['label_audit']={'join_status':merged['_merge'].value_counts().to_dict(),'clinical_tb_status':clinical.tb_status.value_counts().to_dict(),
    'microbiologic_reference':extra.Microbiologicreferencestandard.value_counts(dropna=False).to_dict()}
ref=merged.Microbiologicreferencestandard.map({'Positive':1,'Negative':0})
summary['label_audit']['known_reference_disagreements']=int(((merged.tb_status!=ref)&ref.notna()).sum())
merged[ref.isna() | (merged.tb_status!=ref)].to_csv(OUT/'label_review_required.csv',index=False)
summary['clinical_plausibility_flags']={
    'heart_rate_outside_30_220':clinical.loc[~clinical.heart_rate.between(30,220),'participant'].tolist(),
    'temperature_outside_34_42':clinical.loc[~clinical.temperature.between(34,42),'participant'].tolist(),
    'age_below_18':clinical.loc[clinical.age<18,'participant'].tolist(),
    'height_outside_100_230':clinical.loc[~clinical.height.between(100,230),'participant'].tolist(),
    'weight_outside_25_250':clinical.loc[~clinical.weight.between(25,250),'participant'].tolist()}
archive=Path('dataset.zip')
if archive.exists():
    print('Checking ZIP entries and checksum',flush=True)
    zhash=hashlib.sha256()
    with archive.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):zhash.update(chunk)
    with zipfile.ZipFile(archive) as z:
        entries={x.filename:x for x in z.infolist() if not x.is_dir()}
        mismatch=[]
        for p in all_files:
            key='dataset/'+p.relative_to(ROOT).as_posix();zi=entries.get(key)
            if zi is None or zi.file_size!=p.stat().st_size:mismatch.append(key)
        bad=z.testzip()
    summary['archive']={'bytes':archive.stat().st_size,'sha256':zhash.hexdigest(),'crc_bad_entry':bad,'file_entries':len(entries),'disk_size_mismatches':mismatch}
summary['elapsed_seconds']=time.time()-start
def clean(obj):
    if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,list):return [clean(v) for v in obj]
    if isinstance(obj,np.generic):return obj.item()
    return obj
(OUT/'summary.json').write_text(json.dumps(clean(summary),indent=2,default=str),encoding='utf-8')
print(json.dumps(clean(summary),default=str),flush=True)
