"""Move the models the deployed API needs (DEPLOY.md).

    python app/models_hub.py upload       results/models -> a PRIVATE Hugging Face model repo
    python app/models_hub.py download     that repo -> results/models   (the server, on start)
    python app/models_hub.py pretrained   the public pretrained networks -> the local cache
                                          (Docker build: baked into the image)

upload / download need MODELS_REPO (e.g. your-name/vip-models) and HF_TOKEN (a Hugging Face
access token: "write" to upload, "read" is enough to download).

Why a private repo: the trained models are derived from the ChaLearn First Impressions data,
which is licensed for research. The GitHub repo is public, so they are git-ignored and must not
be committed there.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

import config  # noqa: E402

# Exactly what src/predict.py loads: one classifier + one ridge per lane and trait, and the bundle.
NEEDED = ([f"unimodal_{m}_{t}_{kind}.joblib" for m in ("visual", "audio", "text")
           for t in config.TRAITS for kind in ("clf", "ridge")] + ["bundle.json"])
PRETRAINED = [config.SPEECH_EMOTION_MODEL, config.FACE_EMOTION_MODEL, config.TEXT_EMBED_MODEL,
              config.TEXT_EMOTION_MODEL]
WHISPER_SIZES = ["small.en", "base.en"]        # both baked in, so WHISPER_MODEL can switch at runtime
SKIP_FILES = ["*.h5", "*.msgpack", "*.ot", "tf_model*", "flax_model*", "*.onnx", "onnx/*", "openvino/*",
              "rust_model*", "coreml/*"]


def repo_and_token():
    repo, token = os.environ.get("MODELS_REPO"), os.environ.get("HF_TOKEN")
    if not repo or not token:
        sys.exit("Set MODELS_REPO (e.g. your-name/vip-models) and HF_TOKEN first.")
    return repo, token


def upload():
    from huggingface_hub import HfApi
    repo, token = repo_and_token()
    missing = [f for f in NEEDED if not os.path.exists(os.path.join(config.MODELS_DIR, f))]
    if missing:
        sys.exit(f"missing in results/models: {missing[:3]}... - run ./run_all.sh (or src/06 + src/10) first")
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="model").private:
        sys.exit(f"{repo} exists and is PUBLIC - the models must not be public. Make it private first.")
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=config.MODELS_DIR, allow_patterns=NEEDED,
                      commit_message="trained models for the pipeline visualizer")
    size = sum(os.path.getsize(os.path.join(config.MODELS_DIR, f)) for f in NEEDED) / 1e6
    print(f"uploaded {len(NEEDED)} files ({size:.0f} MB) to the private repo {repo}")


def download():
    from huggingface_hub import snapshot_download
    repo, token = repo_and_token()
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    snapshot_download(repo, repo_type="model", token=token, local_dir=config.MODELS_DIR, allow_patterns=NEEDED)
    missing = [f for f in NEEDED if not os.path.exists(os.path.join(config.MODELS_DIR, f))]
    if missing:
        sys.exit(f"the repo {repo} lacks {missing[:3]}... - run `python app/models_hub.py upload` again")
    print(f"downloaded {len(NEEDED)} model files from {repo}")


def pretrained():
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from torchvision.models import ResNet50_Weights
    for name in PRETRAINED + [f"Systran/faster-whisper-{size}" for size in WHISPER_SIZES]:
        skip = list(SKIP_FILES)
        # Several repos hold the same weights twice; transformers loads the safetensors copy.
        # (wav2vec2 alone: 1.2 GB saved. Whisper's model.bin is its real weights, so not "*.bin".)
        if "model.safetensors" in HfApi().list_repo_files(name):
            skip.append("pytorch_model.bin")
        snapshot_download(name, ignore_patterns=skip)
        print(f"cached {name}")
    torch.hub.load_state_dict_from_url(ResNet50_Weights.IMAGENET1K_V2.url, progress=False)
    print("cached ResNet-50 (IMAGENET1K_V2)")


if __name__ == "__main__":
    actions = {"upload": upload, "download": download, "pretrained": pretrained}
    if len(sys.argv) != 2 or sys.argv[1] not in actions:
        sys.exit(__doc__)
    actions[sys.argv[1]]()
