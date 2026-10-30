"""Stdlib-only offline bootstrap; leaves existing PyTorch unchanged."""
from pathlib import Path
import hashlib,importlib.metadata as md,json,subprocess,sys


def setup(package,install=True):
    package=Path(package)
    def version(name):
        try:return md.version(name)
        except md.PackageNotFoundError:return None
    if version('torch') is None:raise RuntimeError('Choose the existing V4 kernel with PyTorch installed.')
    # V5 introduces two required model libraries; never silently drop a model.
    requested={'librosa':'0.10.2.post1','tabpfn':'2.0.9'}
    if version('torchaudio')!=version('torch'):
        requested['torchaudio']='2.6.0+cu118'
        if version('torch')!='2.6.0+cu118':
            raise RuntimeError(f'Local torchaudio wheel matches torch 2.6.0+cu118, but this kernel has {version("torch")}. Use the successful V4 training kernel. PyTorch will not be replaced.')
    needed=[f'{p}=={v}' for p,v in requested.items() if version(p)!=v]
    if not needed:return {'installed':False,'status':'Required V5 libraries available','python':sys.executable}
    if not install:raise RuntimeError('Missing V5 additions: '+', '.join(needed))
    touched=[p for p in ['numpy','pandas','scipy','sklearn','librosa','tabpfn','torchaudio'] if p in sys.modules]
    if touched:raise RuntimeError('Restart kernel before offline setup; already imported: '+', '.join(touched))
    folder=package/'wheels_v5';checks=json.loads((folder/'checksums.json').read_text())
    for name,expected in checks.items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Local wheel changed: '+name)
    command=[sys.executable,'-m','pip','install','--no-index','--find-links',str(folder),'--disable-pip-version-check',*needed]
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (package/'v5_offline_setup.log').write_text(result.stdout,encoding='utf-8')
    print(result.stdout)
    if result.returncode:raise RuntimeError('Offline installation failed; see research_package/v5_offline_setup.log. Existing torch is preserved.')
    for p,v in requested.items():
        if version(p)!=v:raise RuntimeError('Library verification failed: '+p)
    return {'installed':True,'status':'Offline additions installed before scientific imports','python':sys.executable}
