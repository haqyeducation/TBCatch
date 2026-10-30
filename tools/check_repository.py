"""Check public Git candidates only; never read ignored data or credentials."""
from pathlib import Path
import hashlib,json,re,subprocess

ROOT=Path(__file__).resolve().parents[1]

def main():
    result=subprocess.run(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT,check=True,stdout=subprocess.PIPE)
    names=sorted(set(n.decode('utf-8') for n in result.stdout.split(b'\0') if n))
    tracked=subprocess.run(['git','ls-files','-z'],cwd=ROOT,check=True,stdout=subprocess.PIPE)
    staged_names={n.decode('utf-8') for n in tracked.stdout.split(b'\0') if n}
    errors=[];total=0
    forbidden={'.pem','.key','.p12','.pfx','.pt','.pth','.ckpt','.joblib','.wav','.csv','.zip','.whl','.y4m','.bin','.log'}
    secrets=[re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
             re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),re.compile(r'\bgithub_pat_[A-Za-z0-9_]{30,}\b'),
             re.compile(r'\bhf_[A-Za-z0-9]{30,}\b'),re.compile(r'\bAKIA[A-Z0-9]{16}\b')]
    for name in names:
        p=ROOT/name
        size=p.stat().st_size if p.is_file() else 0;total+=size
        if p.suffix.lower() in forbidden or size>10_000_000 or 'synapse' in p.name.lower() or p.name.startswith('.env'):
            errors.append(f'Private/data/large file eligible for Git: {name}');continue
        if p.suffix.lower() in {'.webp','.woff2'}:continue
        versions=[('working copy',p.read_bytes())] if p.is_file() else []
        if name in staged_names:
            content=subprocess.run(['git','show',':'+name],cwd=ROOT,check=True,stdout=subprocess.PIPE).stdout
            versions.append(('staged',content))
        for location,content in versions:
            if len(content)>10_000_000:errors.append(f'Large {location} file: {name}');continue
            text=content.decode('utf-8-sig',errors='replace')
            if any(pattern.search(text) for pattern in secrets):errors.append(f'Possible credential in {location} {name} (value not printed)')
            if p.suffix=='.ipynb':
                nb=json.loads(text)
                if any(c.get('outputs') or c.get('execution_count') is not None for c in nb['cells'] if c['cell_type']=='code'):
                    errors.append(f'Notebook contains executed outputs: {location} {name}')
    for path in ['prototype/.local/key.pem','prototype/demo.json','prototype/models/bundle.joblib',
                 'data/dataset/dataset/clinical.csv','.venv/pyvenv.cfg','.local/artifacts/experiment_v5_001_RETURN.zip',
                 'research/model_bank_v5/resolved_config.json','Code backup Synapse.txt']:
        if subprocess.run(['git','check-ignore','-q','--no-index',path],cwd=ROOT).returncode!=0:
            errors.append(f'Sensitive path is not ignored: {path}')
    checks=ROOT/'research/source_checksums.json'
    if checks.exists():
        for name,expected in json.loads(checks.read_text())['sha256'].items():
            path='research/'+name
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:errors.append(f'Frozen research source changed: {path}')
            staged=subprocess.run(['git','show',':'+path],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
            if staged.returncode==0 and hashlib.sha256(staged.stdout).hexdigest()!=expected:
                errors.append(f'Staged source bytes differ from V5 provenance (check Git EOL rules): {path}')
    if errors:raise SystemExit('\n'.join(errors))
    print(f'PASS: {len(names)} public files, {total/1_000_000:.2f} MB; no forbidden files, known token patterns or executed notebook outputs.')
    print('Ignored datasets and credential files were not opened. Review the staged diff before publishing.')

if __name__=='__main__':main()
