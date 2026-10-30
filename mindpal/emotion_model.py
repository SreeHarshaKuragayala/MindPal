"""EfficientNet-B0 emotion inference matching the user's uploaded architecture.

Important: the checkpoint tensor shape can confirm the number of outputs, but
cannot prove which human-readable emotion belongs to each output index.
"""
from pathlib import Path
from typing import Any
import json
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

log = logging.getLogger(__name__)

SOURCE_LABELS = [
    "angry", "contempt", "disgust", "fear",
    "happy", "neutral", "sadness", "surprise",
]

def create_efficientnet_b0(num_classes: int = 8) -> nn.Module:
    """Same head layout as the uploaded code; do not change without retraining."""
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model

def _unwrap_state_dict(checkpoint: Any) -> tuple[dict, dict]:
    """Support a raw state_dict or common checkpoint wrappers."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state_dict or a checkpoint dictionary.")
    metadata = {}
    for key in ("class_names", "classes", "class_to_idx", "idx_to_class",
                "labels", "emotion_labels", "metadata", "config"):
        if key in checkpoint:
            metadata[key] = checkpoint[key]
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict) and any(str(k).endswith("classifier.9.weight") for k in candidate):
            checkpoint = candidate
            break
    # Strip DataParallel's prefix only when present on all keys.
    if checkpoint and all(str(k).startswith("module.") for k in checkpoint):
        checkpoint = {str(k)[7:]: v for k, v in checkpoint.items()}
    return checkpoint, metadata

def discover_mapping(metadata: dict, output_count: int):
    """Return mapping from checkpoint metadata when available, otherwise None."""
    raw = (metadata.get("class_names") or metadata.get("classes")
           or metadata.get("labels") or metadata.get("emotion_labels"))
    if isinstance(raw, (list, tuple)) and len(raw) == output_count:
        return [str(x) for x in raw]
    idx = metadata.get("idx_to_class")
    if isinstance(idx, dict):
        try:
            values = [str(idx.get(str(i), idx.get(i))) for i in range(output_count)]
            if all(v != "None" for v in values):
                return values
        except Exception:
            pass
    c2i = metadata.get("class_to_idx")
    if isinstance(c2i, dict) and len(c2i) == output_count:
        try:
            return [str(k) for k, v in sorted(c2i.items(), key=lambda item: int(item[1]))]
        except Exception:
            pass
    return None

class EmotionModel:
    def __init__(self, weights_path: str, mapping_path: str = "config/class_mapping.json",
                 device: str | None = None):
        self.weights_path = Path(weights_path)
        self.mapping_path = Path(mapping_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.class_names = None
        self.mapping_verified = False
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self._load()

    def _load(self):
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {self.weights_path}. Put emotion_model_pytorch.pth in models/."
            )
        # weights_only=True avoids unpickling arbitrary objects where supported.
        try:
            raw = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            raw = torch.load(self.weights_path, map_location="cpu")
        state, metadata = _unwrap_state_dict(raw)
        head = state.get("classifier.9.weight")
        if head is None:
            raise ValueError("Expected classifier.9.weight; checkpoint does not match the uploaded EfficientNet-B0 head.")
        output_count = int(head.shape[0])
        if output_count != 8:
            raise ValueError(f"Expected 8 output classes, checkpoint has {output_count}.")
        self.model = create_efficientnet_b0(output_count)
        try:
            self.model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise ValueError(f"Checkpoint does not exactly match the uploaded EfficientNet-B0 architecture: {exc}") from exc
        self.model.to(self.device).eval()

        embedded = discover_mapping(metadata, output_count)
        config = json.loads(self.mapping_path.read_text(encoding="utf-8")) if self.mapping_path.exists() else {}
        configured = config.get("class_names")
        if embedded:
            self.class_names = embedded
            if configured and list(configured) != embedded:
                raise ValueError(
                    "Checkpoint class metadata conflicts with config/class_mapping.json. "
                    "Resolve the mapping from training artifacts before inference."
                )
            self.mapping_verified = True
            log.info("Class mapping loaded from checkpoint metadata.")
        else:
            if not isinstance(configured, list) or len(configured) != output_count:
                raise ValueError("No valid class mapping. See config/class_mapping.json.")
            self.class_names = [str(x) for x in configured]
            self.mapping_verified = bool(config.get("mapping_confirmed", False))
            if not self.mapping_verified:
                raise ValueError(
                    "The checkpoint confirms 8 outputs but contains no provable label order. "
                    "Check the original training class_to_idx/classes metadata, then set "
                    "mapping_confirmed=true in config/class_mapping.json. Do not guess."
                )
        if len(set(self.class_names)) != output_count:
            raise ValueError("Class names must be unique.")
        log.info("Loaded EfficientNet-B0 on %s with %d classes.", self.device, output_count)

    @torch.inference_mode()
    def predict(self, face_bgr: np.ndarray) -> dict:
        """Return a real model prediction or raise; never synthesize fallback probabilities."""
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Empty face crop.")
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        if logits.ndim != 2 or logits.shape[1] != len(self.class_names):
            raise RuntimeError("Model output shape does not match configured class mapping.")
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        if not np.isfinite(probs).all():
            raise RuntimeError("Model returned non-finite probabilities.")
        idx = int(np.argmax(probs))
        return {
            "emotion": self.class_names[idx],
            "confidence": float(probs[idx]),
            "probabilities": {name: float(probs[i]) for i, name in enumerate(self.class_names)},
        }
