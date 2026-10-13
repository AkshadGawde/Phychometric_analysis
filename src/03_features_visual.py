"""Phase 3 - Visual features.

Both networks are FROZEN feature extractors: model.eval() + torch.no_grad(), no training here.
Each clip has N_FRAMES face crops (Phase 2); a clip's feature is the MEAN over its frames.

1. CNN features (always)
   torchvision ResNet-50 with ImageNet weights (IMAGENET1K_V2). The last layer (`fc`, the
   1000-class ImageNet classifier) is replaced by Identity, so the network returns the
   2048-number description of the image that sits just before it.
   Frames are normalised with the ImageNet mean/std the weights were trained with.
   -> features/visual_cnn.npy            (N, 2048)

2. Face-expression features (only if USE_HEAVY_MODELS)
   Hugging Face ViT `trpakov/vit-face-expression`, with its own image processor.
   (a) softmax over ALL expression classes, in the model's fixed class order
       -> features/visual_emotion.npy    (N, 7)
       -> features/visual_emotion_labels.json  (the class order of the 7 columns)
   (b) CLS embedding = last hidden state[:, 0]
       -> features/visual_vit.npy        (N, 768)

Also writes features/ids.csv: the video_id order (= row order of data/splits.csv) that every
feature file follows. The acceptance check proves each .npy matches it row by row.

Resume: rows computed earlier are reused; only new clips (or clips whose frames were re-made
by Phase 2) go through the networks.

Run:  python src/03_features_visual.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

import utils
import config

log = utils.get_logger("03_visual")


def load_frames(video_id):
    """The N_FRAMES face crops of one clip as RGB PIL images."""
    return [Image.open(utils.frame_path(video_id, k)).convert("RGB") for k in range(config.N_FRAMES)]


def frames_mtime(video_id):
    """When the clip's frames were last written (features older than this are recomputed)."""
    return max(os.path.getmtime(utils.frame_path(video_id, k)) for k in range(config.N_FRAMES))


def mean_over_frames(per_frame, n_clips):
    """(n_clips * N_FRAMES, dim) -> (n_clips, dim): average the frames of each clip."""
    return per_frame.view(n_clips, config.N_FRAMES, -1).mean(dim=1).cpu().numpy()


# ----------------------------------------------------------------------------
# 1. ResNet-50
# ----------------------------------------------------------------------------
def build_resnet(device):
    from torchvision import transforms
    from torchvision.models import ResNet50_Weights, resnet50
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()            # drop the ImageNet classifier -> 2048-d output
    model = model.eval().to(device)           # eval(): fixed batch-norm statistics, no dropout
    # Frames are already IMG_SIZE x IMG_SIZE face crops, so only scale to [0,1] and normalise.
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    @torch.no_grad()
    def compute_batch(batch_ids):
        images = [to_tensor(img) for v in batch_ids for img in load_frames(v)]
        features = model(torch.stack(images).to(device))             # (clips * frames, 2048)
        return {"visual_cnn": mean_over_frames(features, len(batch_ids))}

    return compute_batch


# ----------------------------------------------------------------------------
# 2. ViT face-expression model
# ----------------------------------------------------------------------------
def build_vit(device):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    processor = AutoImageProcessor.from_pretrained(config.FACE_EMOTION_MODEL)
    model = utils.load_pretrained(AutoModelForImageClassification, config.FACE_EMOTION_MODEL)
    model = model.eval().to(device)
    # Fixed class order = the model's own output index order (0, 1, 2, ...).
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    @torch.no_grad()
    def compute_batch(batch_ids):
        images = [img for v in batch_ids for img in load_frames(v)]
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        hidden = model.vit(pixel_values=pixels).last_hidden_state    # (clips * frames, tokens, 768)
        cls = hidden[:, 0]                                           # CLS token = whole-image summary
        # The classifier gives one logit per class; softmax keeps ALL classes (no top-k cut).
        probabilities = torch.softmax(model.classifier(cls), dim=-1)
        return {"visual_emotion": mean_over_frames(probabilities, len(batch_ids)),
                "visual_vit": mean_over_frames(cls, len(batch_ids))}

    return compute_batch, labels


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run_model(label, names, ids, build, device):
    """Load one model, extract its feature files, report timing."""
    start = time.time()
    built = build(device)
    compute_batch, extra = built if isinstance(built, tuple) else (built, None)
    load_sec = time.time() - start
    n_new, sec, replaced = utils.cached_extract(names, ids, compute_batch, frames_mtime,
                                                config.VISUAL_BATCH_CLIPS, label)
    per_clip = f"{sec / n_new:.3f} s per clip" if n_new else "nothing new to compute"
    log.info(f"{label}: model loaded in {load_sec:.1f} s; {n_new} clips in {sec:.1f} s = {per_clip}; "
             f"NaN/inf replaced: {replaced}")
    del built, compute_batch          # drop the model before the next one is loaded (low RAM)
    utils.free_memory()
    return extra


def main():
    utils.set_seed()
    utils.ensure_dirs()
    utils.warn_if_fake_labels()

    df = utils.load_splits()
    ids = df["video_id"].tolist()
    without_frames = [v for v in ids
                      if not all(os.path.exists(utils.frame_path(v, k)) for k in range(config.N_FRAMES))]
    if without_frames:
        sys.exit(f"{len(without_frames)} clip(s) of splits.csv have no frames "
                 f"(e.g. {without_frames[0]}) - run `python src/02_preprocess.py` first.")

    utils.write_ids(ids)                       # the row order every feature file must follow
    device = utils.get_device()
    log.info(f"{len(ids)} clips, device = {device}, USE_HEAVY_MODELS = {config.USE_HEAVY_MODELS}")

    names = ["visual_cnn"]
    run_model("ResNet-50", ["visual_cnn"], ids, build_resnet, device)

    if config.USE_HEAVY_MODELS:
        labels = run_model("ViT face-expression", ["visual_emotion", "visual_vit"], ids,
                           build_vit, device)
        with open(config.VISUAL_EMOTION_LABELS_JSON, "w") as f:
            json.dump(labels, f, indent=2)
        log.info(f"visual_emotion column order: {labels}")
        names += ["visual_emotion", "visual_vit"]
    else:
        log.info("USE_HEAVY_MODELS is off: visual_emotion / visual_vit are not computed")

    # ---- acceptance check: shapes, row order == ids.csv, no NaN ----
    shapes = utils.check_features(names)
    for name, shape in shapes.items():
        log.info(f"  {name}.npy  shape {shape}")
    stale = utils.stale_feature_files(skip=names)
    if stale:
        log.info(f"NOTE: other feature files do not match the current ids.csv and need their "
                 f"script re-run before use: {stale}")
    log.info(f"ACCEPTANCE CHECK PASSED: {len(names)} feature file(s) match features/ids.csv "
             f"({len(ids)} rows, same order), no NaN/inf")
    utils.warn_if_fake_labels()


if __name__ == "__main__":
    main()
