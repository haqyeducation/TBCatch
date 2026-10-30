"""Verify downloaded research inputs; no third-party code is executed."""
from pathlib import Path
import base64,hashlib,io,json,tarfile,zipfile
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parent
def digest(p,kind='sha256'):return hashlib.new(kind,p.read_bytes()).hexdigest()
def destination(folder,name):
    assert not Path(name).is_absolute() and '..' not in Path(name).parts,name
    target=(folder/name).resolve();assert folder.resolve() in target.parents,name
    return target
def write_same(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_bytes()==data,path
    else:path.write_bytes(data)

def main():
    dl=ROOT/'downloads';meta=ROOT/'metadata';manifest=[]
    npm=json.loads((meta/'ppg_js_npm.json').read_text(encoding='utf-8'))
    p=dl/'ppg-js-0.3.0.tgz'
    assert 'sha512-'+base64.b64encode(hashlib.sha512(p.read_bytes()).digest()).decode()==npm['dist']['integrity']
    with tarfile.open(p) as z:
        assert any(n.name=='package/dist/index.global.js' for n in z.getmembers())
        for member in z.getmembers():
            assert member.isfile() or member.isdir(),member.name
            if member.isfile():write_same(destination(ROOT/'vendor/ppg-js-0.3.0',member.name.removeprefix('package/')),z.extractfile(member).read())
    manifest.append(dict(name='ppg-js',version=npm['version'],url=npm['dist']['tarball'],file=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=digest(p),verified='npm SHA-512 integrity; MIT license',scope='Library candidate; no independent clinical validation'))
    p=dl/'welltory_3b506e9.zip';rows=[]
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
        names=[n for n in z.namelist() if n.endswith('/PPG.csv')];assert len(names)==21
        for n in z.namelist():
            if n.endswith('/'):continue
            rel=n.split('/',1)[1]
            write_same(destination(ROOT/'data/welltory',rel),z.read(n))
        for n in names:
            t=pd.read_csv(io.BytesIO(z.read(n)));rr=np.fromstring(z.read(n.replace('PPG.csv','RR.txt')).decode(),sep=' ')
            assert np.isfinite(t[['time','R','G','B']]).all().all() and (np.diff(t.time)>0).all()
            assert len(rr)>0 and np.isfinite(rr).all() and (rr>0).all()
            rows.append(dict(record=n.split('/')[-2],samples=len(t),duration_seconds=(t.time.iloc[-1]-t.time.iloc[0])/1000,rr_count=len(rr),reference_session_bpm=60000/rr.mean()))
    pd.DataFrame(rows).to_csv(meta/'welltory_inventory.csv',index=False)
    manifest.append(dict(name='Welltory',url='https://github.com/Welltory/welltory-ppg-dataset',commit='3b506e9c97c45b89695e87f1d667fe8a033e24fa',file=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=digest(p),verified='ZIP CRC; 21 RGB/timestamp and 21 RR records; CC0 license included',scope='13 volunteers reported; folder names are recordings, not unique participant IDs; no raw videos'))
    fig=json.loads((meta/'figshare_17052410.json').read_text(encoding='utf-8'))
    for f in fig['files']:
        p=dl/f['name'];assert p.stat().st_size==f['size'] and digest(p,'md5')==f['computed_md5']
        manifest.append(dict(name='Fingertip Videos: '+f['name'],url=f['download_url'],doi=fig['doi'],license=fig['license']['name'],file=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=digest(p),verified='Publisher file size and MD5 match'))
    labels=pd.read_excel(dl/'ProjectDatasetReadings.xlsx');assert len(labels)==51 and labels.FileName.is_unique and labels['Heart Rate (bpm)'].notna().all()
    with zipfile.ZipFile(dl/'DatasetVideoFiles.zip') as z:
        assert z.testzip() is None
        videos=[n for n in z.namelist() if n.lower().endswith('.mp4')]
        assert len(videos)==51 and {int(Path(n).stem) for n in videos}==set(labels.FileName)
    labels.to_csv(meta/'figshare_video_labels.csv',index=False)
    # Keep the failed transfer distinguishable from a usable dataset.
    p=dl/'butppg-2.0.0.zip'
    if p.exists() and not zipfile.is_zipfile(p):p.rename(p.with_suffix('.zip.part'))
    sums={line.split(maxsplit=1)[1].strip():line.split()[0] for line in (meta/'butppg_SHA256SUMS.txt').read_text().splitlines()}
    for local,remote in [('butppg_quality-hr-ann.csv','quality-hr-ann.csv'),('butppg_subject-info.csv','subject-info.csv'),('butppg_LICENSE.txt','LICENSE.txt')]:
        assert digest(meta/local)==sums[remote],local
    q=pd.read_csv(meta/'butppg_quality-hr-ann.csv');sub=pd.read_csv(meta/'butppg_subject-info.csv')
    assert len(q)==3888 and q.ID.is_unique and q.Quality.isin([0,1]).all()
    merged=q.merge(sub,on='ID',validate='one_to_one');assert len(merged)==3888
    manifest.append(dict(name='BUT PPG 2.0.0',url='https://physionet.org/content/butppg/2.0.0/',download='https://physionet.org/content/butppg/get-zip/2.0.0/',status='Metadata and license verified against publisher SHA256; full ZIP transfer timed out, NOT available locally',n_records=len(q),n_finger_records=int((merged['Ear/finger']==1).sum()),scope='Processed camera PPG, ECG reference HR and quality labels; includes ear and finger; no raw videos advertised'))
    result=dict(checked_date='2026-10-24',resources=manifest,video_inventory=dict(n_videos=51,n_label_rows=51,reported_unique_participants=24,n_recordings_age18plus=int((labels['Age (years)']>=18).sum()),explicit_participant_mapping=False),scope='Resources verified, not an accuracy benchmark or phone validation')
    (ROOT/'RESOURCE_MANIFEST.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
