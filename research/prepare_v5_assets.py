"""Build pinned local model assets; never runs TB training."""
from pathlib import Path
import hashlib, json, urllib.request

ROOT = Path(__file__).resolve().parent


def fetch(url, path, expected=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print('Downloading', path.name, flush=True)
        tmp = path.with_suffix(path.suffix + '.part')
        with urllib.request.urlopen(url, timeout=90) as src, tmp.open('wb') as dst:
            while block := src.read(1024 * 1024): dst.write(block)
        if expected and hashlib.sha256(tmp.read_bytes()).hexdigest() != expected:
            raise ValueError('Downloaded checksum mismatch: ' + str(path))
        tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and digest != expected: raise ValueError('Asset changed: ' + str(path))
    return {'path':path.relative_to(ROOT).as_posix(), 'sha256':digest,
            'bytes':path.stat().st_size, 'url':url}


def main():
    assets = []
    rev = '732d834db70ee0fc3886b4bcbcfb4ce7fb829be2'
    vendor = ROOT / 'src/vendor_beats'; vendor.mkdir(exist_ok=True)
    (vendor/'__init__.py').write_text('"""Microsoft BEATs; see LICENSE and assets_v5.json."""\n')
    for name in ['BEATs.py', 'backbone.py', 'modules.py']:
        url = f'https://raw.githubusercontent.com/microsoft/unilm/{rev}/beats/{name}'
        data = urllib.request.urlopen(url, timeout=60).read()
        upstream_sha = hashlib.sha256(data).hexdigest()
        data = data.replace(b'from backbone import', b'from .backbone import').replace(b'from modules import', b'from .modules import')
        path = vendor/name; path.write_bytes(data)
        assets.append({'path':path.relative_to(ROOT).as_posix(),'url':url,'revision':rev,
                       'upstream_sha256':upstream_sha,'sha256':hashlib.sha256(data).hexdigest(),
                       'modification':'Relative imports only'})
    assets.append(fetch(f'https://raw.githubusercontent.com/microsoft/unilm/{rev}/LICENSE', vendor/'LICENSE'))
    # Official OneDrive download is unavailable. Two independently hosted mirrors
    # expose this identical LFS hash; do not claim publisher authentication.
    repo = 'lpepino/beats_ckpts'
    info = json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+repo))
    url = f'https://huggingface.co/{repo}/resolve/{info["sha"]}/BEATs_iter3_plus_AS2M.pt'
    entry = fetch(url,ROOT/'pretrained/v5/BEATs_iter3_plus_AS2M.pt',
                  'd43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34')
    entry['provenance'] = 'Third-party mirror; LFS hash also matches mooneyko/BEATs; not authenticated against inaccessible publisher download.'
    assets.append(entry)
    repo = 'Prior-Labs/TabPFN-v2-clf'
    info = json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+repo))
    tree = json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/tree/{info["sha"]}'))
    name = 'tabpfn-v2-classifier-v2_default.ckpt'
    expected = next(r['lfs']['oid'] for r in tree if r['path']==name)
    assets.append(fetch(f'https://huggingface.co/{repo}/resolve/{info["sha"]}/{name}',ROOT/'pretrained/v5'/name,expected))
    assets.append(fetch(f'https://huggingface.co/{repo}/resolve/{info["sha"]}/LICENSE.txt',ROOT/'pretrained/v5/TABPFN_LICENSE.txt'))
    record = {'assets':assets,'beats_upstream_revision':rev,'tabpfn_library':'2.0.9',
              'tabpfn_weights_version':'v2','no_TB_labels_used':True}
    (ROOT/'assets_v5.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('Assets ready',flush=True)


if __name__ == '__main__': main()
