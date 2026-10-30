# Third-party notices

No overall license has been chosen for the original TBCatch project code. Retaining a dependency's license does not license the entire project under the same terms.

- **Microsoft BEATs:** upstream code is vendored under `research/src/vendor_beats/`, with its MIT `LICENSE`. `research/assets_v5.json` records pinned URLs, hashes and relative-import modifications. Model-weight provenance and terms are separate; weights are not committed.
- **Prior Labs TabPFN:** the research uses TabPFN 2.0.9. Its pretrained weights have separate terms, available locally under `research/pretrained/v5/TABPFN_LICENSE.txt`; weights are excluded. The public app does not ship the TabPFN research winner. Built with PriorLabs-TabPFN.
- **8B softlife template:** adapted UI/motion, MIT; `prototype/static/assets/template-LICENSE` and `template-CREDITS.md`. 8B attribution remains in the UI. Original stock photos are not used.
- **GSAP 3.13.0 / ScrollTrigger / SplitText:** GSAP standard license, not the template's MIT license. License URL is retained in the local distribution headers and template credits.
- **Lenis 1.3.11:** MIT; `prototype/static/assets/LENIS-LICENSE.txt`.
- **Kalnia and Onest fonts:** SIL OFL; license files alongside the local font files in `prototype/static/assets/`.
- **ppg-js 0.3.0:** MIT; local browser bundle and license under `prototype/static/vendor/`.
- **Illustrations:** generated using built-in image_gen; prompts/provenance in `prototype/illustration_revision_v2.json`. They are illustrative assets, not patient photos.
- **CODA dataset and research recordings:** not redistributed. Obtain access and follow the source's data-use terms. Embedded demo audio and real cough fixtures are excluded even though they were previously available on this laptop.
- **Other downloaded research references and PPG datasets:** kept locally and excluded from Git. Their respective terms still apply; local availability is not redistribution permission.
