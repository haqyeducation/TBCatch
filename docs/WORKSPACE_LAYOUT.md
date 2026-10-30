# Workspace layout and recovery

Reorganized on 25 October 2026 into `research/`, `prototype/` and local-only `data/`. No experiment archive or dataset was deleted.

- Current V5 source/config files retain their original SHA256 values. Notebook setup paths now discover `research/` and use `data/` as the input root; exported notebook outputs and execution counts were removed from the Git candidates.
- Prototype model/BEATs inference, threshold, frontend behavior and HTTPS certificate remain unchanged. The runtime config is separate from the archived training-machine configuration. PPG browser assets are owned by the prototype.
- `.venv/` retains the existing interpreter/packages. Use its Python explicitly, for example `.venv/Scripts/python.exe -m pip`; do not assume a moved virtual environment is portable to another machine. Fresh clones must build their own environment.
- Historical notebooks, transfer ZIPs, unused assets, previous validation folders, old pretrained PANNs files and offline library collections were moved to the sibling directory **`TB Detection Archive 2026-10-25`**. Its `MOVE_MANIFEST.json` records original and new paths; `cleanup-tools/` contains the reorganization scripts. The Synapse recovery file was moved without opening it, and is not in this Git repository.
- The authorized V5 return still needed for local prototype preparation is under `.local/artifacts/`. Raw CODA data and audit outputs remain under `data/`; trained model banks, run results, reference inputs and offline additions remain local under `research/`. All are ignored by Git.

Public Git candidates are checked by `tools/check_repository.py`: data/model/key extensions, file size, executed notebook outputs and known token patterns. It never scans ignored datasets or credential files. This check is a guardrail, not a guarantee that arbitrary sensitive text cannot be committed; review the staged diff.

Validation after relocation: backend boundary checks, real-model Playwright flow, responsive/reduced-motion checks, research path/asset preflight, unchanged V5 source hashes and fixed-bundle reuse. No TB classifier was retrained as part of cleanup, and full GPU training was not repeated.

No remote, commit or push is created by the cleanup. To publish after reviewing the staged files:

```sh
git diff --cached --stat
python tools/check_repository.py
git commit -m "Initialize TBCatch research and prototype"
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Choose the repository visibility and an overall project license intentionally. Model and dataset access/redistribution terms remain separate from the code repository.
