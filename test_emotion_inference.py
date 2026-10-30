
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mindpal.emotion_model import create_efficientnet_b0
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "models/emotion_model_pytorch.pth"
TEST_DIR = r"F:\Python\Real-Time-Facial-Emotion-Recognition\DATASET\test"

class_names = [
    "angry", "contempt", "disgusted", "fearful",
    "happiness", "neutral", "sadness", "surprised"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
assert dataset.classes == class_names, dataset.classes

loader = DataLoader(dataset, batch_size=32, shuffle=False)

model = create_efficientnet_b0(num_classes=8)
state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state, strict=True)
model.to(device).eval()

all_targets = []
all_predictions = []

with torch.inference_mode():
    for images, targets in loader:
        logits = model(images.to(device))
        predictions = logits.argmax(dim=1).cpu()
        all_targets.extend(targets.tolist())
        all_predictions.extend(predictions.tolist())

correct = sum(
    p == t for p, t in zip(all_predictions, all_targets)
)
accuracy = correct / len(all_targets)

print(f"\nDevice: {device}")
print(f"Test images: {len(all_targets)}")
print(f"Overall accuracy: {accuracy:.2%}")

print("\nPer-class metrics:")
print(classification_report(
    all_targets,
    all_predictions,
    labels=list(range(8)),
    target_names=class_names,
    digits=3,
    zero_division=0
))

print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(
    all_targets,
    all_predictions,
    labels=list(range(8))
))

print("\nEvaluation completed.")
