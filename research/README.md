# TBCatch research — V5

This directory contains the current V5 experiment notebooks, frozen model/research code, asset provenance and verification tests. Earlier standalone notebooks, package-build scripts and validation runs are archived outside the active project. V3/V4 Python modules that remain are transitive dependencies or are protected by the V5 source hashes; do not delete them based on their version names.

## Inputs and layout

```text
research/
  CODA_TB_V5_Research.ipynb       EDA, preprocessing, nested CV and evaluation
  CODA_TB_V5_Resume_XAI.ipynb     Postprocessing-only XAI repair
  config.json, config_v5.json    Original research configuration
  src/                          Frozen source and vendored BEATs code
  prepare_v5_assets.py           Explicit pretrained-asset preparation
  assets_v5.json                 Pinned asset hashes and source provenance
  tools/audit_dataset.py         Read-only dataset audit
  pretrained/v5/                Local weights (ignored)
  wheels_v5/                    Local offline additions (ignored)
  references/                   Local comparison/split references (ignored)
  runs/                         Generated runs and participant results (ignored)
  model_bank_v5/                Local imported trained artifacts (ignored)
  reports/                      Local audit reports/evidence (ignored)
data/
  dataset/dataset/               Authorized original CODA files (ignored)
  dataset-audit/local-full/      Audit manifest and outputs (ignored)
```

Data are not redistributed. Obtain source access and honor data-use terms. The notebook discovers `research/` and defaults its project input root to the sibling `data/` directory. `PACKAGE_DIR` may explicitly select this research directory; `PROJECT_DIR` must select the directory containing `dataset/` and `dataset-audit/` (normally `<project>/data`).

## Environment and running

Use Python 3.11, the tested torch/torchaudio 2.6.0+cu118 pair, tabpfn 2.0.9 and librosa 0.10.2.post1. Install the matching PyTorch build for the training machine first, then the pinned lists in `requirements.txt` and `requirements-v5-additions.txt` using the appropriate PyTorch wheel index or authorized local wheel collection. There is no automatic network fallback in the current notebook; its bootstrap uses local `wheels_v5/` only for missing additions. It does not replace PyTorch.

Pretrained weights are ignored in Git. Restore them with the recorded hashes or explicitly run `prepare_v5_assets.py` after reviewing source/access/license terms. It may require internet and authorized model access. Existing local assets and offline additions remain on this laptop.

Open `CODA_TB_V5_Research.ipynb`. Use `MODE="preflight"` on this laptop. Full training requires the training GPU and respects the 45 GB VRAM limit; never claim that using more memory guarantees better performance. Select a new RUN_NAME for a new scientific configuration. Resume only with matching code, input and environment fingerprints. Notebook outputs and execution counts are cleared for Git; original executed notebooks and returns remain in the external archive/private local storage.

The original research source/config files were preserved byte-for-byte during reorganization. Folder discovery and the notebook's input-root argument changed; no fold, threshold, model-selection rule or archived experiment result was changed. `test_v5_pipeline.py` is an optional synthetic/model-parity check, not TB training; it can take time on CPU. `test_v5_xai_resume.py` covers pointwise XAI repair. Both require research dependencies and assets.

## Research vs prototype

V5 compares full clinical, paper-style audio, frozen BEATs, TabPFN/CatBoost/LR, limited ensembles and phone-available inputs, with participant-disjoint nested evaluation and XAI. The lightweight web app in `../prototype/` uses separate fixed BEATs + CatBoost candidates and an app-only threshold. Its results are not the research winner's performance. No external Indonesia validation or physical-phone pulse validation is established.

Method and historical context: `V5_README.md`, `V5_XAI_FIX_README.md`. Local original return: `../.local/artifacts/experiment_v5_001_RETURN.zip`. Do not commit individual-level exports, raw audio, fitted models or model caches. Refer to `../THIRD_PARTY_NOTICES.md` for third-party licensing boundaries.
