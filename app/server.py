"""The web API of the live pipeline visualizer (ROADMAP_WEB.md, Phase 2).

    uvicorn app.server:app --port 8000          (from the project root, venv active)

POST /api/analyze                  upload an audio / video file -> {job_id}
GET  /api/jobs/{id}/events         Server-Sent Events: the stages of src/predict.py, in order
GET  /api/samples                  recorded demo runs (app/replays/)
GET  /api/samples/{name}/events    a recorded run, streamed again with its original timing
GET  /api/samples/{name}/media/{f} the sample's audio / video, for playback
GET  /api/model-card               the study's results, for the "How it works" page
GET  /api/health                   are the models loaded? how long is the queue?

The models are loaded once, in a background thread, when the server starts. Jobs run one at
a time (the development machine has 8 GB of RAM); waiting jobs get `queued` events.
Privacy: an upload is written to a temporary folder and deleted as soon as its job ends,
also after an error. Nothing of its content is logged or kept.

Deployment (DEPLOY.md): ALLOWED_ORIGINS lets a web app on another domain call this API;
MAX_QUEUE caps the waiting runs.
"""
import asyncio
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import predict  # noqa: E402
import utils  # noqa: E402
import config  # noqa: E402

log = utils.get_logger("server")
WEB_DIST = os.path.join(ROOT, "web", "dist")
JOB_TTL_SEC = 600                     # finished jobs are forgotten after 10 minutes
# Deployed: the web app on another domain (e.g. Vercel) may call this API. Comma-separated list,
# e.g. ALLOWED_ORIGINS=https://my-app.vercel.app ; empty = same origin only (local development).
ALLOWED_ORIGINS = [o.strip().rstrip("/") for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "5"))    # more waiting runs than this -> 429


class Job:
    def __init__(self, path, tmp_dir):
        self.id = uuid.uuid4().hex[:12]
        self.path, self.tmp_dir = path, tmp_dir
        self.events = []
        self.finished = False
        self.created = time.time()
        self.finished_at = None


class Worker:
    """Loads the engine, then runs queued jobs one by one in its own thread."""

    def __init__(self):
        self.engine = None
        self.status = "starting"
        self.error = None
        self.jobs = {}
        self.waiting = []                       # job ids in queue order
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    @property
    def ready(self):
        return self.status == "ready"          # models loaded AND warmed up

    def submit(self, job):
        with self.lock:
            self.jobs[job.id] = job
            self.waiting.append(job.id)
        self.queue.put(job.id)

    def position(self, job_id):
        with self.lock:
            return self.waiting.index(job_id) if job_id in self.waiting else -1

    def _load(self):
        try:
            self.engine = predict.Engine()

            def progress(name):
                self.status = f"loading {name}"
                log.info(self.status)
            t0 = time.time()
            self.engine.load(progress)
            progress("warm-up run")
            self.engine.warm_up()
            self.status = "ready"
            log.info(f"models loaded in {time.time() - t0:.0f} s on {self.engine.device}")
        except BaseException as error:              # SystemExit from a missing bundle, too
            self.status, self.error = "failed", str(error)
            log.error(f"loading the models failed: {error}")

    def _loop(self):
        self._load()
        while True:
            job_id = self.queue.get()
            job = self.jobs.get(job_id)
            if job is None:
                continue
            try:
                if not self.ready:
                    job.events.append({"type": "error", "stage": "ingest",
                                       "message": f"the models could not be loaded: {self.error}"})
                else:
                    for event in self.engine.run(job.path):
                        job.events.append(event)
            except Exception as error:              # noqa: BLE001 - reported to the browser
                log.exception("job failed")
                job.events.append({"type": "error", "stage": "ingest", "message": str(error)})
            finally:
                shutil.rmtree(job.tmp_dir, ignore_errors=True)      # the upload is gone now
                with self.lock:
                    if job_id in self.waiting:
                        self.waiting.remove(job_id)
                job.finished, job.finished_at = True, time.time()
                self._forget_old()

    def _forget_old(self):
        now = time.time()
        with self.lock:
            for job_id in [j for j, job in self.jobs.items()
                           if job.finished and now - job.finished_at > JOB_TTL_SEC]:
                del self.jobs[job_id]


worker = Worker()
app = FastAPI(title="VIP pipeline visualizer")
if ALLOWED_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["*"])


def sse(event):
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n"


@app.get("/api/health")
def health():
    return {"ready": worker.ready, "status": worker.status, "error": worker.error,
            "device": str(worker.engine.device) if worker.engine else None,
            "queue": len(worker.waiting), "whisper": config.WHISPER_MODEL}


@app.post("/api/analyze")
async def analyze(file: UploadFile):
    if len(worker.waiting) >= MAX_QUEUE:
        raise HTTPException(429, f"{len(worker.waiting)} runs are already waiting - try again in a minute")
    tmp_dir = tempfile.mkdtemp(prefix="vip-")
    ext = os.path.splitext(file.filename or "")[1].lower()[:6] or ".bin"
    path = os.path.join(tmp_dir, "input" + ext)
    size, limit = 0, config.APP_MAX_UPLOAD_MB * 1024 * 1024
    with open(path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                out.close()
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise HTTPException(413, f"the file is larger than {config.APP_MAX_UPLOAD_MB} MB")
            out.write(chunk)
    if size == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, "the file is empty")
    job = Job(path, tmp_dir)
    worker.submit(job)
    return {"job_id": job.id, "position": worker.position(job.id)}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    job = worker.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown or expired job")

    async def stream():
        sent, last_status = 0, None
        while True:
            if await request.is_disconnected():
                return
            if sent == 0 and not job.finished and not job.events:
                position = worker.position(job_id)
                status = {"type": "queued", "position": position, "ready": worker.ready,
                          "status": worker.status}
                if status != last_status:
                    yield sse(status)
                    last_status = status
            while sent < len(job.events):
                yield sse(job.events[sent])
                sent += 1
            if job.finished and sent >= len(job.events):
                return
            await asyncio.sleep(0.1)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ----------------------------------------------------------------------------
# Replays (demo mode)
# ----------------------------------------------------------------------------
def load_replay(name):
    path = os.path.join(config.REPLAYS_DIR, f"{os.path.basename(name)}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "unknown sample")
    with open(path) as f:
        return json.load(f)


@app.get("/api/samples")
def samples():
    out = []
    if os.path.isdir(config.REPLAYS_DIR):
        for name in sorted(os.listdir(config.REPLAYS_DIR)):
            if name.endswith(".json"):
                replay = load_replay(name[:-5])
                done = replay["events"][-1]
                out.append({"name": replay["name"], "title": replay["title"],
                            "description": replay.get("description", ""),
                            "audio": replay.get("audio"), "video": replay.get("video"),
                            "seconds": done.get("seconds")})
    return out


@app.get("/api/samples/{name}/events")
async def sample_events(name: str, request: Request, speed: float = 1.0):
    replay = load_replay(name)
    speed = min(max(speed, 0.25), 20.0)

    async def stream():
        start = time.time()
        for event in replay["events"]:
            due = event.get("t", event.get("seconds", 0)) / speed
            wait = due - (time.time() - start)
            if wait > 0:
                await asyncio.sleep(wait)
            if await request.is_disconnected():
                return
            yield sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/samples/{name}/media/{filename}")
def sample_media(name: str, filename: str):
    replay = load_replay(name)
    if filename not in (replay.get("audio"), replay.get("video")):
        raise HTTPException(404, "unknown file")
    return FileResponse(os.path.join(config.REPLAYS_DIR, filename))


# ----------------------------------------------------------------------------
# Model card
# ----------------------------------------------------------------------------
@app.get("/api/model-card")
def model_card():
    with open(config.BUNDLE_JSON) as f:
        bundle = json.load(f)
    return JSONResponse({
        **bundle["model_card"],
        "traits": bundle["traits"], "trait_labels": predict.TRAIT_LABELS,
        "models": bundle["models"], "feature_sets": bundle["feature_sets"],
        "subsets": {k: {t: v["test"] for t, v in per_trait.items()} for k, per_trait in bundle["fusion"].items()},
        "weights": bundle["weights"], "default_choice": bundle["default_choice"],
        "held_out_clips": bundle["held_out_clips"], "whisper": config.WHISPER_MODEL,
        "created_at": bundle["created_at"], "git_commit": bundle["git_commit"],
    })


# The built frontend (npm run build) is served from the same port; in development Vite serves it.
if os.path.isdir(WEB_DIST):
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
