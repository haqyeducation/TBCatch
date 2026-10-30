"""Build an explicitly unpaired flow demo from existing local research inputs."""
from pathlib import Path
import base64,csv,io,json,wave,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
fold=json.loads((ROOT/'research/model_bank_v5/nested_splits.json').read_text())['folds'][0]
with (ROOT/'data/dataset-audit/local-full/solicited_training_manifest_with_review_flags.csv').open(newline='',encoding='utf-8') as f:
    rows=[r for r in csv.DictReader(f) if r['participant']==fold['test'][0]][:3]
clips=[]
for r in rows:
    path=ROOT/'data/dataset/dataset'/r['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
    with wave.open(str(path)) as w:
        assert w.getnchannels()==1 and w.getsampwidth()==2;sr=w.getframerate();clips.append(w.readframes(w.getnframes()))
data=(b'\0\0'*int(sr*.65)).join(clips);buf=io.BytesIO()
with wave.open(buf,'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr);w.writeframes(data)
samples=[]
with (ROOT/'research/heart_rate_resources/data/welltory/data/subject_01/PPG.csv').open() as f:
    for row in csv.DictReader(f):samples.append([float(row['time'])/1000,float(row['R']),float(row['G']),float(row['B'])])
demo=dict(audio=base64.b64encode(buf.getvalue()).decode(),audio_seconds=len(data)/(sr*2),ppg_samples=samples,
          clinical_demo=dict(age=35,sex=0,reported_cough_dur=21,tb_prior=0,hemoptysis=0,weight_loss=0,smoke_lweek=0,fever=0,night_sweats=0),
          notice='Unpaired technical demo: held-out cough examples plus unrelated PPG and fictional questionnaire. Not a patient or accuracy experiment.')
(ROOT/'prototype/demo.json').write_text(json.dumps(demo,separators=(',',':')),encoding='utf-8')
(ROOT/'prototype/tests/cough_example.wav').write_bytes(buf.getvalue())
silent=io.BytesIO()
with wave.open(silent,'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b'\0\0'*16000)
(ROOT/'prototype/tests/silence.wav').write_bytes(silent.getvalue())
print('Demo audio, real PPG and test fixtures prepared. No patient labels or clinical records exported.')
