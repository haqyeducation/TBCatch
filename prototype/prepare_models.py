"""Fit two fixed V5 candidates for a CPU MVP, without selecting on outer-test."""
from pathlib import Path
import io,json,hashlib,sys,zipfile
import joblib,numpy as np
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];PACKAGE=ROOT/'research';OUT=Path(__file__).resolve().parent/'models'
sys.path.insert(0,str(PACKAGE))
from src import coda_v5 as v5,coda_v5_models as models

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'bundle.joblib').exists():
        manifest=json.loads((OUT/'manifest.json').read_text())
        assert hashlib.sha256((OUT/'bundle.joblib').read_bytes()).hexdigest()==manifest['bundle_sha256']
        print('Existing verified CPU MVP bundle reused.');return
    with zipfile.ZipFile(ROOT/'.local/artifacts/experiment_v5_001_RETURN.zip') as z:
        checks=json.loads(z.read('run/artifact_checksums.json'))
        def read(p):
            b=z.read('run/'+p);assert hashlib.sha256(b).hexdigest()==checks[p],p
            return b
        cfg=json.loads(read('resolved_config.json'));fold=json.loads(read('nested_splits.json'))['folds'][0]
        for name,h in cfg['source_hashes'].items():assert hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()==h,name
        cfg.update(package_root=str(PACKAGE),cpu_threads=4)
        table=joblib.load(io.BytesIO(read('participant_features.joblib')));grid={c['id']:c for c in models.candidate_grid(cfg)}
        bundles={};calibrated={}
        with threadpool_limits(limits=4):
            for name in ['beats_hr','beats_mobile']:
                candidate=grid[name+'_02'];assert candidate['family']=='cb'
                rows=[]
                for j in range(5):
                    r=json.loads(read(f'outer_00/inner/{candidate["id"]}_{j}.json'))
                    assert r['train_ids']==fold['inner_folds'][j]['train'] and r['validation_ids']==fold['inner_folds'][j]['validation']
                    rows.extend(zip(r['validation_ids'],r['raw_probability'],r['y']))
                rows.sort();assert {r[0] for r in rows}==set(fold['proper_train'])
                assert all(int(table.at[pid,'tb_status'])==y for pid,p,y in rows)
                iso=IsotonicRegression(y_min=0,y_max=1,increasing=True,out_of_bounds='clip').fit([r[1] for r in rows],[r[2] for r in rows])
                print('Preparing fixed CPU candidate',candidate['id'],flush=True)
                model=models.fit_model(table,fold['proper_train'],candidate,cfg,cfg['seed']+99)
                cal=table.loc[fold['calibration']];raw=models.predict_raw(model,cal,cfg);p=iso.predict(raw)
                singleton=models.predict_raw(model,cal.iloc[[0]],cfg)[0];assert abs(singleton-raw[0])<1e-10
                calibrated[name]=p
                bundles[name]=dict(model=model,calibrator=iso,candidate=candidate,proper_ids=fold['proper_train'],calibration_ids=fold['calibration'])
        y=table.loc[fold['calibration'],'tb_status'].to_numpy()
        gate=v5.threshold(y,calibrated['beats_hr'],'sensitivity90',.99)
        route=calibrated['beats_hr']>=gate;p2=calibrated['beats_mobile'];eligible=[np.nextafter(p2.max(),np.inf),*np.unique(p2)[::-1],0.]
        valid=[]
        for cut in eligible:
            d=route&(p2>=cut)
            if d[y==1].mean()>=.9:valid.append((float((~d[y==0]).mean()),float(cut)))
        assert valid
        cascade=dict(gate=gate,stage2_cutoff=max(valid)[1])
        # Prototype artifacts stay separate from every V5 training/evaluation artifact.
        joblib.dump(dict(bundles=bundles,cascade=cascade),OUT/'bundle.joblib',compress=3)
        manifest=dict(purpose='CPU MVP, fixed existing CatBoost candidates; NOT V5 selected_full winner or a validated deployment model',
            candidates={k:b['candidate'] for k,b in bundles.items()},outer_fold=0,
            selection='Fixed trial index 02 for each output, chosen for CPU feasibility; no outer-test scores used',
            source_archive_sha256=hashlib.sha256((ROOT/'.local/artifacts/experiment_v5_001_RETURN.zip').read_bytes()).hexdigest(),
            source_protocol=cfg['protocol_fingerprint'],cascade=cascade,proper_n=len(fold['proper_train']),calibration_n=len(fold['calibration']),
            bundle_sha256=hashlib.sha256((OUT/'bundle.joblib').read_bytes()).hexdigest(),clinical_validation=False)
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('CPU MVP bundle ready.',flush=True)

if __name__=='__main__':main()
