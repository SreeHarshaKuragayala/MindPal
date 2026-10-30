"""Inspect checkpoint keys and any embedded class metadata without running the camera."""
import argparse
import json
from pathlib import Path
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", default="models/emotion_model_pytorch.pth")
    args = p.parse_args()
    path = Path(args.checkpoint)
    if not path.is_file():
        raise SystemExit(f"Not found: {path}")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    print("Top-level type:", type(obj).__name__)
    if not isinstance(obj, dict):
        print("This is not a state-dict/checkpoint dictionary.")
        return
    metadata_keys = ("class_names", "classes", "class_to_idx", "idx_to_class",
                     "labels", "emotion_labels", "metadata", "config")
    print("Metadata:")
    for key in metadata_keys:
        if key in obj:
            print(f"  {key}: {json.dumps(obj[key], default=str)}")
    state = obj
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(obj.get(key), dict):
            state = obj[key]
            break
    for key in ("classifier.9.weight", "module.classifier.9.weight"):
        if key in state:
            print(f"Output count from {key}: {int(state[key].shape[0])}")
            break
    else:
        print("Could not find classifier.9.weight. This may be a different checkpoint format.")
    print("\nNote: output count is not the same as label-order verification.")

if __name__ == "__main__":
    main()
