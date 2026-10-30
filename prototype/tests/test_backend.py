"""Boundary checks only; real model inference is covered by browser integration."""
from pathlib import Path
import base64,io,sys,unittest,wave
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inference import read_wav,segment_audio,validate_pulse,validate_clinical,APP_THRESHOLD,screening_category,screening_guidance

def wav(x,channels=1,sr=16000):
    b=io.BytesIO()
    with wave.open(b,'wb') as w:w.setnchannels(channels);w.setsampwidth(2);w.setframerate(sr);w.writeframes(np.asarray(x,dtype='<i2').tobytes())
    return base64.b64encode(b.getvalue()).decode()

class BoundaryTests(unittest.TestCase):
    def test_both_screening_results_include_follow_up(self):
        for flagged in [False,True]:
            advice=screening_guidance(flagged)
            self.assertIn('not a diagnosis',advice['heading'])
            self.assertIn('negative result cannot rule out TB',advice['limitation'])
            self.assertIn('sputum rapid molecular test',advice['testing'])
            self.assertIn('clinical assessment',advice['next_step'])
            self.assertTrue(advice['source_url'].startswith('https://www.who.int/'))
        self.assertNotEqual(screening_guidance(True)['next_step'],screening_guidance(False)['next_step'])
    def test_app_threshold_boundary(self):
        self.assertGreater(APP_THRESHOLD,.13186813186813187)
        self.assertEqual(screening_category(APP_THRESHOLD-1e-9),'research_screening_negative')
        self.assertEqual(screening_category(APP_THRESHOLD),'research_screening_positive')
        self.assertEqual(screening_category(.271),'research_screening_positive')
    def test_wav_silence_rejected(self):
        with self.assertRaises(ValueError):read_wav(wav(np.zeros(16000)))
    def test_stereo_rejected(self):
        with self.assertRaises(ValueError):read_wav(wav(np.ones(32000)*1000,channels=2))
    def test_clipping_rejected(self):
        with self.assertRaises(ValueError):read_wav(wav(np.ones(16000)*32767))
    def test_invalid_wav(self):
        for v in ['not-base64',base64.b64encode(b'not a wave').decode(),None]:
            with self.assertRaises(ValueError):read_wav(v)
    def test_real_audio_segmentation_bounded(self):
        f=Path(__file__).parent/'cough_example.wav';x,sr=read_wav(base64.b64encode(f.read_bytes()).decode());clips,duration=segment_audio(x,sr)
        self.assertGreater(duration,1);self.assertGreaterEqual(len(clips),1);self.assertLessEqual(len(clips),12)
        self.assertTrue(all(0<len(c)<=8000 and np.isfinite(c).all() for c in clips))
    def test_missing_or_invalid_pulse(self):
        for v in [None,{},dict(bpm=0,source='manual'),dict(bpm=float('nan'),source='manual'),dict(bpm=True,source='manual'),dict(bpm=80,source='unknown')]:
            with self.assertRaises(ValueError):validate_pulse(v)
    def test_camera_needs_quality_evidence(self):
        with self.assertRaises(ValueError):validate_pulse(dict(bpm=80,source='camera',quality_passed=False))
        self.assertEqual(validate_pulse(dict(bpm=80,source='camera',quality_passed=True,capture_seconds=30,good_windows=3)),(80,'camera'))
    def test_clinical_requires_complete_adult_inputs(self):
        good=dict(age=35,sex=0,reported_cough_dur=21,tb_prior=0,hemoptysis=0,weight_loss=0,smoke_lweek=0,fever=0,night_sweats=0)
        self.assertEqual(len(validate_clinical(good)),9)
        for values in [{},dict(good,age=12),dict(good,fever=True),dict(good,reported_cough_dur=-1),dict(good,sex=3),dict(good,age='35')]:
            with self.assertRaises(ValueError):validate_clinical(values)

if __name__=='__main__':unittest.main(verbosity=2)
