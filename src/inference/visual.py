"""Phase 3 building blocks: face crops -> ResNet-50 features + ViT face-expression features.

Moved here from src/03_features_visual.py (the behaviour is unchanged) so that src/predict.py
uses exactly the same steps. Both builders return a function that takes a flat list of RGB PIL
images (N_FRAMES per clip, clip after clip) and returns PER-FRAME outputs; the caller averages
them per clip with mean_over_frames(). See the docstring of src/03_features_visual.py.
"""
import torch

import utils
import config


def mean_over_frames(per_frame, n_clips):
    """(n_clips * N_FRAMES, dim) -> (n_clips, dim): average the frames of each clip."""
    return per_frame.view(n_clips, config.N_FRAMES, -1).mean(dim=1).cpu().numpy()


def build_resnet(device):
    """Returns embed(images) -> tensor (n_images, 2048)."""
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
    def embed(images):
        return model(torch.stack([to_tensor(img) for img in images]).to(device))

    return embed


def build_vit(device):
    """Returns (analyse, labels). analyse(images) -> (probabilities (n, 7), CLS embeddings (n, 768))."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    processor = AutoImageProcessor.from_pretrained(config.FACE_EMOTION_MODEL)
    model = utils.load_pretrained(AutoModelForImageClassification, config.FACE_EMOTION_MODEL)
    model = model.eval().to(device)
    # Fixed class order = the model's own output index order (0, 1, 2, ...).
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    @torch.no_grad()
    def analyse(images):
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        hidden = model.vit(pixel_values=pixels).last_hidden_state    # (images, tokens, 768)
        cls = hidden[:, 0]                                           # CLS token = whole-image summary
        # The classifier gives one logit per class; softmax keeps ALL classes (no top-k cut).
        return torch.softmax(model.classifier(cls), dim=-1), cls

    return analyse, labels
