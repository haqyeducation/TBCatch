# TBCatch prototype

Buildless HTML/CSS/JS with a Python standard-library HTTP server. The phone measures pulse locally; cough audio is analyzed on the laptop by frozen BEATs and two fixed CatBoost candidates. This is an investigational prototype, not the V5 selected TabPFN model or a clinically validated diagnostic app.

## Existing laptop

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\prototype\start.ps1
# Phone access over local HTTPS:
powershell -ExecutionPolicy Bypass -File .\prototype\start.ps1 -Phone
```

The model interpreter defaults to `.venv/Scripts/python.exe`; override with `-PythonExe` when needed. The HTTPS certificate helper uses `python` on PATH, with cryptography installed. Existing certificates in `prototype/.local/` are reused. Open http://localhost:8765 on the laptop. Use the HTTPS LAN address printed by the helper on a phone connected to the same network. Self-signed certificate acceptance and browser permissions may be necessary. Do not publish this local standard-library server directly to the internet.

If Windows Firewall blocks phone access, run `prototype/allow_phone_access.ps1` manually as Administrator. It permits TCP 8443 from LocalSubnet, without disabling the firewall. The legacy rule ID `NapasMvpHttpsLocal` is retained to avoid duplicates. To revoke it: `Remove-NetFirewallRule -Name 'NapasMvpHttpsLocal'`.

## Clean clone prerequisites

No model, dataset, demo recording, private certificate, Python environment or test output is included in Git.

1. Create a Python 3.11 environment and install a matching torch 2.6.0 / torchaudio pair for the target platform, then `pip install -r prototype/requirements.txt`. Research/training dependencies are separate; see `research/README.md`.
2. Restore authorized pretrained assets under `research/pretrained/v5/` and verify `research/assets_v5.json`. `research/prepare_v5_assets.py` can fetch the pinned resources; review the URLs, access conditions and license terms first. It is not run automatically by the app.
3. Restore the trusted fitted `prototype/models/bundle.joblib` matching `prototype/models/manifest.json`. It remains available locally on this laptop. Do not load untrusted pickle/joblib files. Alternatively, an authorized V5 return archive at `.local/artifacts/experiment_v5_001_RETURN.zip` allows `prototype/prepare_models.py` to fit the same fixed MVP candidates; this preparation is not the original nested-CV training.
4. Restore `prototype/demo.json` and `prototype/tests/cough_example.wav` only where their data-use terms permit. `prototype/prepare_demo.py` rebuilds these local fixtures from authorized `data/` and `research/model_bank_v5/` inputs; it contains no embedded patient audio.
5. For HTTPS, install cryptography in the Python on PATH and use `start.ps1 -Phone` to create local certificates. Never commit the generated keys.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s prototype/tests -v
python prototype/tests/make_camera_fixture.py
node prototype/tests/e2e.cjs
node prototype/tests/design.cjs
```

Playwright and Chrome must be installed. Browser test scripts use local Playwright if available, with this laptop's bundled runtime as fallback. Tests expect the HTTPS app on localhost:8443 (override BASE_URL). Real cough fixtures/demo data are local prerequisites; the camera Y4M is generated synthetically. Simulated 72 BPM footage checks the acquisition path, not accuracy on people.

## Privacy, result interpretation and design

Raw audio and questionnaire answers are not written to files. Up to four audio embeddings stay in server memory; session data is not stored in localStorage. Explicit summary downloads include the screening result and follow-up guidance, without the audio payload. The demo combines unrelated cough and pulse samples with fictional answers and is labelled throughout. Same-origin/token/host checks and input/quality validation remain active. There is no account authentication; only use a trusted local test network.

App-only final threshold: 18.18/100. Initial questionnaire gate: zero, so all initial checks request the questionnaire. A positive result does not confirm TB; a negative result cannot rule it out. The camera has not been validated against a reference pulse device. See [UI revision and follow-up guidance](UI_REVISION_2.md), [asset prompts](illustration_revision_v2.json), and the aggregate threshold audit in `models/app_threshold_audit.json`.
