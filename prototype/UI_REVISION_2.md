# TBCatch UI revision 2 — 24 October 2026

The 8B template typography, palette and section transitions remain. Copy now describes inputs and actions directly: Start screening, Measure your resting pulse, Record your cough, and Complete the health questionnaire. Therapeutic welcome language has been removed. An inline SVG brand spark uses currentColor, preventing emoji substitution on phones.

## Five illustrations and five different motions

All assets were created with built-in **image_gen** before integration in the same cream, burgundy, crimson, pink and powder blue cartoon style. Full prompts, correction prompts and original generated filenames: [illustration_revision_v2.json](illustration_revision_v2.json). Pillow only converted PNG to WebP. These are raster illustrations animated through CSS/GSAP, not generated videos or patient photographs.

| Placement | File in static/assets | Motion |
|---|---|---|
| Hero, woman coughing into a tissue | hero-cough-v2-cartoon.webp | Paired small cough movements |
| Step 1, man measuring resting pulse | pulse-step-v2-cartoon.webp | Settle and scale entrance |
| Step 2, man coughing into a tissue while recording | record-cough-v2-cartoon.webp | Different sideways cough movement |
| Step 3, woman completing health questionnaire | context-v2-cartoon.webp | Horizontal reveal |
| Close-up grip instructions | pulse-grip-v2-cartoon.webp | Brief alignment tilt |

The grip follows the user's AVIF reference: palm-facing view, index extended behind the phone, thumb and three fingers supporting it, screen facing the user. The step illustration was corrected so the index fingertip covers the camera. The close-up uses a transparent phone to explain the grip; its caption asks users to adjust it to their actual camera position. Do not cough during pulse measurement.

Each placement has a unique image and animation. Offscreen motion pauses. Repeating movements have finite iteration counts. Reduced motion disables image and scroll animations and leaves content visible. The existing template dome, wordmark exit and arch reveal remain; the three step illustrations now also have different scroll entrances.

## Result guidance

Every final questionnaire result includes `guidance` from `inference.screening_guidance()`, rendered in the browser and included in the downloaded summary. Both categories state that this is initial screening from an investigational prototype, not a diagnosis. Positive does not confirm TB; negative cannot rule it out.

Positive results recommend a clinic/Puskesmas assessment and discussion of further TB testing. Negative results still recommend assessment for persistent/worsening symptoms or close TB contact. A healthcare professional determines appropriate testing, which may include a sputum rapid molecular test and chest imaging.

General follow-up advice was checked against [WHO screening Q&A](https://www.who.int/news-room/questions-and-answers/item/systematic-screening-for-tb) and [WHO TB fact sheet](https://www.who.int/news-room/fact-sheets/detail/tuberculosis), accessed 24 October 2026. Source links appear in the UI. These sources do not endorse or validate TBCatch.

## Verification and unchanged model

- Backend: 10 checks, including threshold equality and guidance for both categories.
- Playwright e2e: 19 checks with real model inference, guidance rendering, recording/upload, validation, reset, API protection and simulated media lifecycle.
- Playwright design: 375/768/1440 px, five unique assets and animations, SVG mark, informative copy, dome/manifesto motion, navigation, overflow, asset loads, JavaScript errors and reduced motion.

Evidence: `test-results/`. Dedicated `tbcatch-*` viewport captures show each section at its relevant scroll position.

Unchanged: fixed BEATs + CatBoost bundle, initial questionnaire gate of zero, final app threshold of 2/11 (18.18/100), and original V5 research. Calibration observations are not independent test results or validated phone-app performance. Existing HTTPS certificate and firewall policy are reused.

Not verified: physical phone pulse accuracy against a reference instrument or clinical diagnostic performance.
