# TBCatch

Research and a local mobile-web prototype for initial TB screening using cough audio, estimated pulse and reported symptoms. **Investigational research only; not a diagnosis or a validated clinical screening product.**

The research compares frozen BEATs and paper-style audio features with clinical inputs, patient-disjoint nested cross-validation and XAI. The prototype uses fixed lightweight **BEATs + CatBoost** candidates. It does **not** deploy the research's selected TabPFN winner, and research scores must not be attributed to the app.

## Structure

```text
research/           V5 notebooks, frozen source/config, asset preparation and tests
prototype/          Python server, inference, web UI, launcher and browser tests
data/               Local CODA audio, clinical metadata and audit outputs (ignored)
.venv/              Existing local Python 3.11 environment (ignored)
.local/             Private V5 return archive and cleanup records (ignored)
tools/              Repository readiness check
```

Model weights, fitted bundles, participant-level records, datasets, notebook outputs, credentials, certificates, offline wheels and generated test evidence are excluded from Git. A clone contains code and UI assets; it does not contain a ready-trained medical model or permission to redistribute CODA data.

## Run locally on this laptop

```powershell
powershell -ExecutionPolicy Bypass -File .\prototype\start.ps1
```

Open [http://localhost:8765](http://localhost:8765). To enable the existing local HTTPS phone workflow:

```powershell
powershell -ExecutionPolicy Bypass -File .\prototype\start.ps1 -Phone
```

Keep the phone and laptop on the same network and use the HTTPS LAN address printed by the certificate helper. The self-signed certificate warning is expected for this local prototype. See [prototype setup](prototype/README.md) for permissions, firewall scope and clean-clone prerequisites.

## Research

Open [the V5 notebook](research/CODA_TB_V5_Research.ipynb) with a compatible Python 3.11 GPU environment. The notebook resolves `research/` and `data/` from the new layout. It defaults to full training; use `MODE="preflight"` for local checks. Do not run full GPU training on this laptop. See [research setup](research/README.md).

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s prototype/tests -v
node prototype/tests/e2e.cjs
node prototype/tests/design.cjs
python tools/check_repository.py
```

Browser tests require a running HTTPS server, Chrome, Playwright and the local demo fixtures. The synthetic camera fixture tests lifecycle and signal handling; it does not validate pulse accuracy on people.

## GitHub preparation

The local repository uses `main`. Before a commit, run `python tools/check_repository.py` and review `git diff --cached --stat`. No remote is configured and nothing is pushed automatically. Do not force-add ignored data or credentials. An overall project license has not been selected; third-party licenses are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Limitations

This is an initial research screening. A positive result does not confirm TB and a negative result cannot rule it out. Clinical assessment and appropriate testing remain necessary. Camera pulse estimates have not been validated against a reference device. The app threshold is 18.18/100; its calibration tradeoff is documented in [the current UI notes](prototype/UI_REVISION_2.md).
