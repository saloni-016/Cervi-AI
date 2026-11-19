# CerviAI — Repository guide

CerviAI is a Flask prototype for explainable cervical screening support. It runs an XGBoost model on a clinical questionnaire, a Vision Transformer (ViT) on a cervical-cell image, or both models side by side. Flask renders the interface with Jinja templates and serves CSS and vanilla JavaScript; no separate frontend build is required.

**Scope:** The scores, heatmaps, and AI explanations are educational. They do not diagnose or rule out cervical cancer and should not be used to make clinical decisions.

## Repository structure

```text
Cervi-AI/
├── app.py                     # Flask entry point, routes, sessions, PostgreSQL/SQLite, Groq
├── explain_xgboost.py          # Clinical inference and SHAP explanations
├── train_xgboost.py            # Train and evaluate the clinical model
├── predict_clinical.py        # Interactive clinical prediction in the terminal
├── model.pkl                  # Saved XGBoost classifier
├── imputer.pkl                # Saved mean imputer for the clinical features
├── train_vit.py               # ViT checkpoint loader and inference helpers (not training)
├── explain_vit.py             # Image inference, occlusion heatmap, and region scores
├── predict_vit.py             # Interactive single-image classification
├── check_vit.py               # Evaluate the checkpoint on a SIPaKMeD split
├── vit_sipakmed.pth           # ViT weights managed by Git LFS
├── sipakmed_classes.json      # Ordered class labels for the ViT output
├── templates/                 # Landing, sign-in, assessment, result, and history pages
├── static/                    # UI assets; legacy uploads/heatmaps excluded from deploy
├── frontend/                  # Static GitHub Pages landing page source
├── requirements.txt           # Local Python dependencies, including shared requirements
├── requirements-base.txt      # Shared dependencies, including the PostgreSQL driver
├── Dockerfile                 # CPU-only Linux container serving Gunicorn
├── .github/workflows/         # Manual Railway container and GitHub Pages workflows
├── deploy/                    # Deployment helpers, Neon schema provisioning, Pages build
└── instructions.txt           # Step-by-step deployment guide
```

The main app serves [`templates/index.html`](templates/index.html) at `/`. [`frontend/index.html`](frontend/index.html) is the static GitHub Pages landing template. After publishing, `https://saloni-016.github.io/Cervi-AI/` links to the configured HTTPS Railway service. It does not render Flask forms, call APIs or use database credentials.

## How an assessment flows through the app

```text
/ → /signup or /signin → /choose
                           ├─ /details             → POST /predict          → /result
                           ├─ /image_upload        → POST /predict_image    → /image_result
                           └─ /combined_assessment → POST /predict_combined → /combined_result

Clinical-only and combined submissions → clinical score summary in PostgreSQL or SQLite → /history
```

On startup, [`app.py`](app.py) loads environment variables and clinical model artifacts, checks database access with `init_db()`, and then initializes a temporary media directory. SQLite tables are created locally if needed. For PostgreSQL, [`deploy/prepare_neon.py`](deploy/prepare_neon.py) provisions the tables once with the owner; the runtime receives only a restricted login with read and insert grants. Railway requires that pooled Neon URL with verified TLS and channel binding, `SECRET_KEY`, and `DEMO_PASSWORD`; it never falls back to SQLite. Signup or sign-in puts the user's email and username in the Flask session. The ViT weights are loaded on the first image inference, not at app startup.

### Clinical assessment

1. `predict()` reads 12 numeric questionnaire fields in the order expected by the model. [`explain_xgboost.py`](explain_xgboost.py) defines that feature order; keep it aligned with the forms and training script when changing inputs.
2. `explain_clinical_input()` applies the saved imputer, runs XGBoost, and returns the class-1 (`Biopsy` positive) probability as a percentage. The app labels it **Low Risk** below 40, **Medium Risk** from 40 to below 70, and **High Risk** at 70 or above. These are app-defined score bands, not validated clinical thresholds.
3. SHAP explains which features pushed the model output higher or lower. The helper normalizes absolute SHAP values across all 12 features and returns the top five. Those influence percentages describe model behavior, not causation or medical risk. `format_feature_value()` and `create_clinical_explanation()` turn the factors into readable text.
4. `predict()` saves the score summary on submission, holds the detailed assessment temporarily on the worker and redirects to `/result`. Refreshing the result page does not create another history row.

### Cell image analysis

1. `predict_image()` accepts a multipart upload named `scan_image` (`.bmp`, `.png`, `.jpg`, or `.jpeg`), validates the image format and dimensions, saves it under a random ID in a temporary directory, and calls `explain_image()`.
2. [`train_vit.py`](train_vit.py) provides `get_model()`, which loads and caches a ViT-B/16 checkpoint on CPU or CUDA, and `predict_risk()`, a helper for inference without a heatmap. The image is converted to RGB, resized to 224 × 224, and normalized before inference. The classifier predicts one of five cell types: Dyskeratotic, Koilocytotic, Metaplastic, Parabasal, or Superficial-Intermediate.
3. [`explain_vit.py`](explain_vit.py) uses `get_probabilities()` for batched inference. Confidence is the top class's softmax probability. The abnormality score is the sum of the Dyskeratotic and Koilocytotic probabilities. `risk_level()` labels that sum **Low** below 0.40, **Moderate** from 0.40 to below 0.70, and **High** at 0.70 or above. The standalone result page displays **Medium** instead of **Moderate**.
4. `explain_image()` masks one 16 × 16 patch at a time and measures how much the target score drops. It explains the abnormality score if it is at least 0.40; otherwise it explains the predicted class. It saves a PNG overlay and returns up to three contributing regions. The heatmap shows model sensitivity, not confirmed abnormal tissue.
5. `predict_image()` stores the latest output on the worker and keeps an opaque reference in the session. `/image_result` displays the indicator, class, image, and heatmap. `/full_report` shows the classification confidence and an HTML explanation. Image-only runs do not create history rows.

### Combined assessment and assistant

`predict_combined()` requires both the 12 numeric fields and an image. It runs the same clinical and image functions and keeps their detailed outputs temporarily on the worker. It records a mismatch flag if one model's indicator is high and the other's is low, but the current template does not render that flag. **The models remain separate; there is no fused medical risk score.**

If `GROQ_API_KEY` is set, the combined route sends model summaries (including selected clinical feature values) to Groq's `openai/gpt-oss-120b` for a plain-language interpretation. If the key is unset or the call fails, the page still shows the two model outputs with fallback text. Only the clinical score and category are written to history for a combined assessment.

The floating assistant in [`templates/chatbot.html`](templates/chatbot.html) and [`static/chatbot.js`](static/chatbot.js) sends questions to `POST /clinical-result-chat`. The clinical result page has its own inline widget that includes the displayed score and feature explanations as context. The endpoint handles simple greetings locally and otherwise sends the question and supplied context to the same Groq model. Neither Groq workflow sends the raw image file.

## Flask route and function map

| Route | Handler | What it does |
| --- | --- | --- |
| `GET /` | `home()` | Renders the landing page. |
| `GET /healthz` | `healthz()` | Returns a minimal startup check without requiring credentials. |
| `GET/POST /signup`, `GET/POST /signin` | `signup()`, `signin()` | Render account forms, create an account or verify credentials, and start a session. |
| `GET /choose`, `GET/POST /logout` | `choose()`, `logout()` | Show the dashboard or confirm sign-out, then clear the session on POST. |
| `GET /details`, `POST /predict`, `GET /result` | `details()`, `predict()`, `result()` | Show the questionnaire, run the clinical model, and display/save the result. |
| `GET /image_upload`, `POST /predict_image`, `GET /image_result` | `image_upload()`, `predict_image()`, `image_result()` | Show the upload form, run image analysis, and render the image result. |
| `GET /api/image_result` | `api_image_result()` | Returns the latest standalone image class, confidence, and asset URLs as JSON. Any clinical score in this response comes from a separate clinical assessment in the session, not from the image model. |
| `GET /assessment-media/<media_id>/<filename>` | `assessment_media()` | Serves an unexpired image or heatmap only if it belongs to the current assessment session. |
| `GET /full_report` | `full_report()` | Renders the latest standalone image report as HTML. |
| `GET /combined_assessment`, `POST /predict_combined`, `GET /combined_result` | `combined_assessment()`, `predict_combined()`, `combined_result()` | Collect both inputs, run both models, and present their results side by side. |
| `GET /history`, `GET /compare` | `history()`, `compare()` | List saved scores for the signed-in user or calculate the change between the latest two. `/compare` cannot render because `templates/compare.html` is missing. |
| `POST /clinical-result-chat` | `clinical_result_chat()` | Accepts JSON with `question` and optional `result_context`; returns JSON with `answer`. |

Assessment, report, history and chat routes require a signed-in session. `/api/image_result` returns `401` without one and `404` if there is no standalone image result. When `DEMO_PASSWORD` is set, a separate invitation prompt covers app and asset requests except `/healthz`. Forms and chat require a CSRF token; private routes are not cached by the browser.

## Where data lives

| Location | What it holds |
| --- | --- |
| Neon PostgreSQL or local `database.db` | `users` stores usernames, emails, and password hashes for new accounts. `results` stores the user's email, clinical score category, probability, and timestamp. History is newest-first by row ID; it does not reconstruct full reports. |
| Flask session cookie | Login state, CSRF token and opaque result references. Detailed assessment inputs/outputs are not stored in the cookie. |
| Bounded worker memory | The latest detailed reports; they expire and are lost on instance restart. |
| Temporary worker directory | Recent originals and generated overlays, not persisted to PostgreSQL. Media URLs are session-restricted; media can expire after an hour, be evicted on another upload, or be lost across instances. Old files in `static/uploads/` and `static/explanations/` remain tracked but are not served by Flask or copied into the container. |

## Run locally

Work from the repository root. Use Python 3.12 and install the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a local `.env` file with your own values. It is ignored by Git; do not commit it.

```dotenv
SECRET_KEY=replace-with-a-long-random-value
# GROQ_API_KEY=optional-for-ai-text
```

`SECRET_KEY` enables signed Flask sessions. Leave `DATABASE_URL` unset to use SQLite. Only set `GROQ_API_KEY` if you want the optional chat and combined interpretation; the models run without it. Old plaintext accounts in an existing SQLite database are not migrated or accepted for sign-in.

For image-only or combined assessments, install [Git LFS](https://git-lfs.com/) and fetch the real checkpoint (about 343 MB). Without LFS the checkout can contain only a text pointer, which PyTorch cannot load; standalone clinical inference does not use that checkpoint.

```bash
git lfs install
git lfs pull
```

Start the local server with `python -m flask --app app run` and open <http://127.0.0.1:5000>. For a smoke check, create a test account, submit synthetic clinical inputs, and confirm the result appears in History. With the checkpoint available, use a non-patient demo image you are authorized to use and check that the heatmap renders. Then try the combined flow and confirm the two scores are displayed separately.

## Deploy to Railway with GitHub Actions and Neon

Follow [`instructions.txt`](instructions.txt) for the first setup. Deploy a fresh image from the Railway-enabled workflow; older images do not enable hosted safeguards on Railway.

1. **GitHub Actions → GHCR.** `Publish CerviAI demo` verifies the Git LFS checkpoint, builds the nonroot CPU image, audits installed Python dependencies, and smoke-tests Gunicorn and the model artifacts. It publishes a distinct tag per build and records the registry digest. Run it on the default branch with deployment off the first time.
2. **Railway → Neon.** Create a Railway Docker Image service from that tag, add the pooled app-only Neon URL and the two app secrets as Railway service variables, and configure `/healthz`, one replica and HTTPS. Once the service is working, opt in to the Railway deploy job for later releases. It uses an environment-scoped token, deploys that run's tested tag and checks the new deployment and public invitation gate. Railway must be able to pull the GHCR package; don't make it public without confirming redistribution rights. Measure image size and peak inference memory before relying on a small plan; paid usage can incur charges.
3. **GitHub Pages.** Set the `CERVIAI_APP_URL` repository variable to the generated `https://…up.railway.app` origin and run `Publish CerviAI landing page`. The default `https://saloni-016.github.io/Cervi-AI/` site has no server or secrets. Visitors leave Pages to sign in on the Railway host. No custom domain is required.

The workflows are manual and run from the default branch. A green publish job alone does not prove a live deployment. `/healthz` verifies startup, not full image inference; use synthetic data for the hosted checks in the guide. Removing the repo's Vercel configuration does not disconnect an existing Vercel project.

Run `python -m unittest discover -s tests -v` after installing the app dependencies for offline checks of hosted settings, the Pages artifact, the Railway deployment API flow, sessions, and history. They use stubs, not live Railway or Neon credentials. Actions also loads the real model artifacts before pushing.

## Offline training and evaluation

The datasets are not bundled. Run these scripts from the repository root; the Flask app does not retrain either model during requests.

| Command | Inputs and outputs |
| --- | --- |
| `python train_xgboost.py` | Reads `kag_risk_factors_cervical_cancer.csv` from the repo root and uses `Biopsy` as the target. It takes a stratified 80/20 split, fits the mean imputer on training data only, and weights the positive class to address imbalance. It prints accuracy, a classification report, and a confusion matrix, then overwrites `model.pkl` and `imputer.pkl`. |
| `python predict_clinical.py` | Prompts for 12 numeric features and prints the prediction, positive-class probability, and top SHAP factors using the saved artifacts. |
| `python predict_vit.py` | Prompts for a local image path and prints the predicted class and confidence. Requires the checkpoint and class-label JSON. |
| `python check_vit.py` | Reads an `ImageFolder` dataset at `../sipakmed_images/` (a sibling of the repo, organized by class). Class indices must match `sipakmed_classes.json`. It evaluates the saved checkpoint on the test portion of a stratified 70/15/15 split and prints a confusion matrix and classification report. |

Despite its name, `train_vit.py` does not train the ViT; the training pipeline for that checkpoint is not included. Use `explain_image()` or the Flask upload flow to generate a heatmap: the `explain_vit.py` command-line entry point currently only prints usage.

## Prototype limits

- **Use demo data only.** New passwords are hashed, and signed cookies carry session metadata, not full reports. Any legacy plaintext rows still in a local database are unchanged. Clinical text can be sent to Groq if enabled. Files already in Git history are not made private by the deployment allowlists.
- **Some pages are unfinished.** The comparison handler exists but has no template. `admin.html`, `chat.html`, and `education.html` are not wired to Flask routes and reference a missing `base.html`. The active assistant is the floating widget, not the unused `chat.html` page.
- **Hosting is for a small invitation-only demo.** Railway deployments hold detailed results and images on a worker, not in durable storage. The endpoint can receive unauthenticated requests even though the app requires an invitation. There is no distributed rate limiting. Check CPU inference time and cold-start latency after deploying. This is not a production service for patient data.
