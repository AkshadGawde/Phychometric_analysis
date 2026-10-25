# Deploying the pipeline visualizer

```
 browser ──► Vercel (web/, static React app)
    │
    └──────► Render (Docker: app/server.py + all models, CPU)  ◄── private Hugging Face repo
                                                                     (the 31 trained model files)
```

- **Vercel** serves the web app: static files, free tier is enough.
- **Render** runs the API in Docker. The image already contains the public pretrained networks
  (wav2vec2, ViT, ResNet-50, MiniLM, the text-emotion model, Whisper).
- **Your trained models** (43 MB) live in a **private** Hugging Face repo and are downloaded when
  the server starts. They are not in GitHub: the repo is public and the models are derived from
  ChaLearn data (research licence).

## What it needs (measured with the image in Docker, 2 CPUs)

| | |
|---|---|
| Memory | **peak 2.0 GB** during a run, ~0.8 GB idle → use a **4 GB** instance; 2 GB runs out |
| Speed | 15 s video ≈ **34 s**, voice-only ≈ **29 s** (CPU, `WHISPER_MODEL=base.en`) |
| Start-up | ≈ 50 s from container start to `"ready": true` |
| Image | ≈ 5 GB uncompressed, first build ≈ 10 min |

**Render's free instance (512 MB) cannot load the models.** `render.yaml` asks for the `pro` plan
(4 GB). Check the current price at render.com/pricing before you deploy: it is a monthly cost.
**Free alternative for the API:** a Hugging Face *Docker Space* (free CPU tier with far more RAM).
The same Dockerfile works there; see the end of this file.

---

## 1. Push the code to GitHub

Render and Vercel both build from your GitHub repo.

```bash
cd ~/Desktop/Developer/VIP
git status          # check: no recordings/, no results/models/, no data/ (all git-ignored)
git add -A
git commit -m "Live pipeline visualizer + calibration + deployment"
git push
```

## 2. Upload the trained models to a private Hugging Face repo (once)

1. Make a free account at huggingface.co.
2. Settings → Access Tokens → create a token with **write** access (for the upload).
3. Upload. The script creates the repo as private, and refuses if it exists and is public:

```bash
source .venv/bin/activate
MODELS_REPO=<your-hf-name>/vip-models HF_TOKEN=hf_xxx python app/models_hub.py upload
```

4. Create a second token with **read** access only. That is the one Render gets.

## 3. Deploy the API on Render

1. render.com → **New → Blueprint** → connect GitHub → pick this repo. Render reads `render.yaml`.
2. It asks for the secret values:
   - `MODELS_REPO` = `<your-hf-name>/vip-models`
   - `HF_TOKEN` = the **read** token
   - `ALLOWED_ORIGINS` = leave empty for now (step 5)
3. Deploy. The first build takes ~10 minutes.
4. Check: `https://<service-name>.onrender.com/api/health`. After about a minute it shows `"ready": true`.

## 4. Deploy the web app on Vercel

1. vercel.com → **Add New → Project** → import the same GitHub repo.
2. **Root Directory: `web`** (important). Framework: Vite (detected).
3. Environment variable: `VITE_API_BASE` = `https://<service-name>.onrender.com` (no trailing `/`).
4. Deploy. You get `https://<project>.vercel.app`.

Or from the terminal: `cd web && vercel --prod` (it asks the same questions; add the variable with
`vercel env add VITE_API_BASE production`, then deploy again).

`VITE_API_BASE` is built into the files. If you change it, redeploy on Vercel.

## 5. Let the web app talk to the API

On Render: your service → **Environment** → `ALLOWED_ORIGINS` = `https://<project>.vercel.app`
(exactly as the browser shows it, no trailing `/`; several origins: comma-separated). Save, and
Render restarts the service.

## 6. Test

Open `https://<project>.vercel.app`:
- the status dot top-right turns green ("models ready · cpu");
- *Demo samples* plays a recorded run;
- *Record voice* / *Record video* work: the browser allows the microphone on https;
- *How it works* shows the study's numbers.

---

## Updating

- **Code:** `git push`. Vercel and Render rebuild by themselves.
- **Models** (after re-running `./run_all.sh`): `python app/models_hub.py upload` again, then on
  Render: *Manual Deploy → Restart*. The models are downloaded at every start.
- **Demo samples:** `./app/make_samples.sh` (or `--record` your own), commit `app/replays/`, push.

## When something goes wrong

| Symptom | Cause / fix |
|---|---|
| Render logs end with `Killed` / exit 137 | out of memory: the plan has less than 4 GB |
| Web app says "API offline" | the service is still starting (~1 min), or `VITE_API_BASE` is wrong: redeploy Vercel after fixing it |
| Browser console: CORS error | `ALLOWED_ORIGINS` does not match the Vercel URL exactly |
| Start-up fails: "Set MODELS_REPO and HF_TOKEN" / 401 | missing or wrong token, or the token cannot read the private repo |
| A run takes > 1 min | normal on 1 CPU; use ≥ 2 CPUs, keep `OMP_NUM_THREADS` = number of CPUs |
| "N runs are already waiting" (429) | `MAX_QUEUE` reached: one run at a time per instance |

Test the image locally before pushing (Docker Desktop running):

```bash
docker build -t vip-api .
docker run --rm -p 8000:8000 --cpus 2 -e WHISPER_MODEL=base.en \
  -v "$PWD/results/models:/app/results/models:ro" vip-api      # mounts the models: no HF repo needed
# then: curl localhost:8000/api/health
```

## Alternative: the API on a Hugging Face Space (free CPU)

1. huggingface.co → New Space → SDK **Docker** → private or public.
2. Push this repo to the Space (or add the Space as a second git remote). In the Space's `README.md`
   front matter add `app_port: 8000`.
3. Space settings → Secrets: `MODELS_REPO`, `HF_TOKEN`, `ALLOWED_ORIGINS`; variables
   `WHISPER_MODEL=base.en`, `OMP_NUM_THREADS=2`.
4. On Vercel, `VITE_API_BASE` = the Space's direct URL (`https://<user>-<space>.hf.space`).

Free Spaces sleep when idle: the first visit after a while waits for the start-up.

---

## Before you share the link

This app films faces and records voices and then makes judgments about people. Before the link goes
beyond you and your supervisor:

- **Consent and data:** the server deletes uploads after each run and logs nothing, but the
  recordings still pass through Render (and Vercel serves the page). Faces and voices count as
  biometric / personal data under GDPR. Say so on the page (the consent box already does) and do
  not collect anything.
- **Dataset licence:** the models are trained on ChaLearn First Impressions, released for research.
  Check that a public demo of models trained on it fits its terms, or keep the link private.
- **Use:** apparent-personality and emotion recognition are what the EU AI Act restricts in hiring,
  workplaces and education. The app says it is a research prototype; keep it that way.
- **Cost / abuse:** anyone with the link uses your paid instance. Share it privately, or put the Vercel
  project behind Vercel's password protection, and keep `MAX_QUEUE` low.
