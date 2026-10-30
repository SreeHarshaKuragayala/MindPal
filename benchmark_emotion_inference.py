
import time
import torch
from mindpal.emotion_model import create_efficientnet_b0

MODEL_PATH = "models/emotion_model_pytorch.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_efficientnet_b0(num_classes=8)
state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state, strict=True)
model.to(DEVICE).eval()

# Simulate one preprocessed 224x224 RGB face image.
sample = torch.randn(1, 3, 224, 224, device=DEVICE)

# Warm-up runs; CUDA needs time to initialize.
with torch.inference_mode():
    for _ in range(20):
        model(sample)

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

runs = 100
start = time.perf_counter()

with torch.inference_mode():
    for _ in range(runs):
        model(sample)

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Average model inference: {elapsed / runs * 1000:.2f} ms")
print(f"Throughput: {runs / elapsed:.2f} images/sec")
