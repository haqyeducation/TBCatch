"""Local prototype inference. Archived research artifacts are never modified."""
from pathlib import Path
import base64,binascii,hashlib,io,json,math,sys,threading,time,wave
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'research'
sys.path.insert(0,str(PACKAGE))
from src import coda_v5_models as models,coda_v5_features as features

FIELDS=models.MOBILE
# App-only operating point: next distinct calibration score above 0.131868.
# shortcut: calibration observations are not phone-app validation; validate prospectively before deployment.
APP_THRESHOLD=2/11

def screening_category(score):
    return 'research_screening_positive' if score>=APP_THRESHOLD else 'research_screening_negative'

def screening_guidance(flagged):
    return dict(
        heading='Initial screening, not a diagnosis',
        limitation='TBCatch is an investigational research prototype. A positive result does not confirm TB, and a negative result cannot rule out TB.',
        next_step=('Contact a clinic or Puskesmas for a clinical assessment and discuss further TB testing.' if flagged else
                   'Seek a clinical assessment if your cough persists, symptoms worsen, or you have been in close contact with someone with TB, even with this negative screening result.'),
        testing='A more definite assessment requires a healthcare professional and appropriate tests, which may include a sputum rapid molecular test and chest imaging.',
        symptoms='Persistent cough, fever, night sweats, or unexplained weight loss should be assessed regardless of this score.',
        source_url='https://www.who.int/news-room/questions-and-answers/item/systematic-screening-for-tb',
        source_label='WHO: TB screening and further testing')

def read_wav(encoded):
    if not isinstance(encoded,str) or len(encoded)>8_000_000:raise ValueError('Audio exceeds the size limit. Use a recording up to 30 seconds.')
    try:
        payload=base64.b64decode(encoded,validate=True)
        with wave.open(io.BytesIO(payload)) as w:
            sr=w.getframerate();frames=w.getnframes()
            if w.getnchannels()!=1 or w.getsampwidth()!=2 or w.getcomptype()!='NONE':raise ValueError('Audio must be mono 16-bit PCM WAV.')
            if not 8000<=sr<=96000 or not .25<=frames/sr<=30:raise ValueError('Audio must last 0.25–30 seconds at a supported sample rate.')
            data=w.readframes(frames)
        if len(data)!=frames*2:raise ValueError('This audio file is incomplete.')
        x=np.frombuffer(data,dtype='<i2').astype(np.float32)/32768
    except (wave.Error,EOFError,binascii.Error) as exc:raise ValueError('We could not read this WAV. Choose another file or record again.') from exc
    if not np.isfinite(x).all() or float(np.sqrt(np.mean(x*x)))<.001:raise ValueError('Audio is too quiet or empty. Record your cough again, closer to your phone.')
    if float(np.mean(abs(x)>=.999))>.05:raise ValueError('Audio is distorted. Move your phone a little farther away and record again.')
    return x,sr

def segment_audio(x,sr):
    """Energy windows are a prototype, not a medically validated cough detector."""
    import librosa
    from scipy.signal import find_peaks
    x=librosa.resample(x,orig_sr=sr,target_sr=16000) if sr!=16000 else x
    x=np.asarray(x,np.float32)
    if len(x)<=9600:return [x[:8000]],len(x)/16000
    hop=160;window=320
    rms=np.array([np.sqrt(np.mean(x[i:i+window]**2)) for i in range(0,len(x)-window+1,hop)])
    cutoff=max(.002,float(rms.max())*.15)
    peaks,_=find_peaks(rms,height=cutoff,distance=65,prominence=max(.001,float(rms.max())*.10))
    if not len(peaks):peaks=np.array([int(rms.argmax())])
    peaks=sorted(sorted(peaks,key=lambda i:rms[i],reverse=True)[:12])
    clips=[]
    for peak in peaks:
        # shortcut: energy onset approximates prepared CODA crops; validate real recordings before clinical use.
        onset=int(peak)
        while onset>0 and peak-onset<30 and rms[onset-1]>max(cutoff,rms[peak]*.20):onset-=1
        start=max(0,onset*hop-160)
        clips.append(x[start:start+8000])
    return clips,len(x)/16000

def validate_pulse(pulse):
    if not isinstance(pulse,dict):raise ValueError('Measure your pulse first.')
    bpm=pulse.get('bpm')
    if isinstance(bpm,bool) or not isinstance(bpm,(int,float)) or not math.isfinite(bpm) or not 45<=bpm<=220:raise ValueError('Pulse is missing or outside the prototype range (45–220 BPM).')
    source=pulse.get('source')
    if source not in ['camera','manual','demo']:raise ValueError('Unknown pulse measurement source.')
    if source=='camera':
        try:valid=pulse.get('quality_passed') is True and float(pulse.get('capture_seconds',0))>=25 and int(pulse.get('good_windows',0))>=3
        except (ValueError,TypeError):valid=False
        if not valid:raise ValueError('Pulse quality is insufficient. Please measure again.')
    return float(bpm),source

def validate_clinical(values):
    if not isinstance(values,dict) or set(values)!=set(FIELDS):raise ValueError('Please answer all nine questions.')
    result={}
    for name in FIELDS:
        value=values[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Your answers are incomplete or contain an invalid number.')
        if name=='age':valid=18<=value<=100
        elif name=='reported_cough_dur':valid=0<=value<=3650
        else:valid=value in [0,1]
        if not valid:raise ValueError('Check your age, cough duration, and answers.')
        result[name]=float(value)
    return result

class Predictor:
    def __init__(self):
        self.phase='Preparing the local model';self.error=None;self.ready=False
        self.lock=threading.Lock();self.bundles={};self.net=None;self.cache={}

    def status(self):return dict(ready=self.ready,phase=self.phase,error=self.error,model='BEATs + CatBoost · lightweight V5 prototype',inference='CPU, one participant per prediction',clinical_validation=False)

    def initialize(self):
        try:
            import torch,joblib
            from src import coda_research as cr
            cr.seed_everything(42);torch.set_num_threads(4)
            self.cfg=json.loads((Path(__file__).resolve().parent/'runtime_config.json').read_text())
            self.cfg.update(package_root=str(PACKAGE),tabpfn_device='cpu',cpu_threads=4,require_cuda_for_full=False)
            self.phase='Verifying model assets';features.verify_assets(PACKAGE)
            self.phase='Loading the audio encoder';self.net=features.load_beats(self.cfg,'cpu')
            path=Path(__file__).resolve().parent/'models/bundle.joblib'
            manifest=json.loads(path.with_name('manifest.json').read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['bundle_sha256']:raise RuntimeError('MVP model checksum mismatch.')
            saved=joblib.load(path);self.cascade=saved['cascade'];self.bundles=saved['bundles']
            for i,name in enumerate(['beats_hr','beats_mobile'],1):
                self.phase=f'Checking model {i}/2'
                t=self.table(np.zeros((1,768),np.float32),80,{k:35 if k=='age' else 14 if k=='reported_cough_dur' else 0 for k in FIELDS})
                first=self.score(name,t);second=self.score(name,t)
                if abs(first[0]-second[0])>1e-8:raise RuntimeError('Model predictions are inconsistent. Model activation stopped.')
            self.phase='Local model ready';self.ready=True
        except Exception as exc:
            import traceback
            traceback.print_exc();self.error='Model could not load. Check the server terminal for details.';self.phase='Model unavailable'

    @staticmethod
    def table(bag,bpm,clinical):
        row=dict(clinical,heart_rate=bpm,_bag_beats=bag)
        return pd.DataFrame([row],index=['mvp_session'])

    def score(self,name,table):
        assert len(table)==1
        b=self.bundles[name];raw=float(models.predict_raw(b['model'],table,self.cfg)[0])
        probability=float(np.clip(b['calibrator'].predict([raw])[0],0,1))
        if not math.isfinite(probability):raise RuntimeError('Model returned an invalid score.')
        return raw,probability

    def analyze(self,request):
        if not self.ready:raise RuntimeError('The model is loading. Wait for the ready status above.')
        bpm,source=validate_pulse(request.get('pulse'))
        stage=request.get('stage')
        if stage not in ['initial','questionnaire']:raise ValueError('Unknown screening stage.')
        clinical=validate_clinical(request.get('clinical')) if stage=='questionnaire' else {}
        encoded=request.get('audio')
        x,sr=read_wav(encoded)
        key=hashlib.sha256(encoded.encode()).hexdigest();started=time.perf_counter()
        if key in self.cache:bag,seconds,nclips=self.cache[key]
        else:
            clips,seconds=segment_audio(x,sr)
            bag=features.embed_batch(self.net,clips,self.cfg,'cpu');nclips=len(clips)
            # Small in-memory cache only; audio and patient responses are never written to disk.
            if len(self.cache)>=4:self.cache.pop(next(iter(self.cache)))
            self.cache[key]=(bag,seconds,nclips)
        name='beats_hr' if stage=='initial' else 'beats_mobile'
        raw,p=self.score(name,self.table(bag,bpm,clinical))
        cut=self.cascade['gate'] if stage=='initial' else APP_THRESHOLD
        flagged=p>=cut
        return dict(stage=stage,score=p,raw_score=raw,threshold=cut,above_threshold=flagged,
            category='context_needed' if stage=='initial' else screening_category(p),
            threshold_policy='initial_questionnaire_gate' if stage=='initial' else 'app_calibration_next_score',
            original_calibration_threshold=self.cascade['stage2_cutoff'] if stage=='questionnaire' else None,
            questionnaire_recommended=flagged if stage=='initial' else None,
            title='Complete the health questionnaire' if stage=='initial' else ('Research screening positive' if flagged else 'Research screening negative'),
            message=('Your cough and pulse have been analyzed. Add nine answers to see your full research screening result.' if stage=='initial' else
                     'Your model score meets the app screening threshold. This does not confirm TB. Speak with a healthcare professional about your symptoms.' if flagged else
                     'Your model score is below the app screening threshold. This cannot rule out TB. Persistent symptoms still need a healthcare assessment.'),
            guidance=screening_guidance(flagged) if stage=='questionnaire' else None,
            demo=source=='demo' or request.get('audio_source')=='demo',pulse_source=source,
            audio_seconds=round(seconds,2),audio_windows=nclips,elapsed_seconds=round(time.perf_counter()-started,2),
            model=name,model_family='BEATs + CatBoost',candidate=self.bundles[name]['candidate']['id'],checkpoint='mvp_fixed_v5_candidates',inference='CPU pointwise',validated_for_clinical_use=False)
