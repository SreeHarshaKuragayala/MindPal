# MindPal — Stage 1

Stage 1 validates the existing EfficientNet-B0 model locally with a webcam. It is intentionally separate from cloud AI, speech, and Raspberry Pi integration.

## Included

- The exact EfficientNet-B0 classifier head structure from your uploaded scripts.
- Image preprocessing from the scripts: RGB, 224×224, ImageNet mean/std.
- Webcam face detection using OpenCV's bundled Haar cascade.
- Fail-closed loading and inference: missing/broken weights or an unverified label mapping stops the app; no random/mock predictions.
- `inspect_checkpoint.py` to inspect whether your checkpoint embeds class labels.
- A consent-gated local identity registry scaffold for later face-recognition enrollment.

## Important label-order limitation

Your uploaded scripts define the eight labels as:

`angry, contempt, disgust, fear, happy, neutral, sadness, surprise`

That establishes the mapping used by the scripts, **not necessarily the order used during training**. A plain `state_dict` usually contains output weights but not the semantic class names. Do not set `mapping_confirmed` to `true` until you verify the order from the original training code/dataset `class_to_idx`, saved checkpoint metadata, or training logs. If the checkpoint includes metadata, the loader uses it and checks for conflicts.

## Setup (Windows PowerShell)

1. Install Python 3.10 or 3.11 (recommended).
2. Open this folder in VS Code / terminal.
3. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   Install the PyTorch/Torchvision build appropriate to your operating system from the official PyTorch installer if the default install is unsuitable.

4. Copy your weights file to:

   `models/emotion_model_pytorch.pth`

5. Inspect checkpoint metadata:

   ```powershell
   python inspect_checkpoint.py
   ```

6. Confirm label order from the original training artifacts. Edit `config/class_mapping.json` only after that verification, setting `mapping_confirmed` to `true` and retaining the verified ordered list.

7. Run:

   ```powershell
   python app.py
   ```

   Press **Q** or **Esc** to exit. If you have multiple cameras, try `python app.py --camera 1`.

## macOS / Linux

Create a virtual environment with `python3 -m venv .venv`, activate it, install `requirements.txt`, place the weights file in `models/`, verify mapping, and run `python app.py`.

## Stage 1 acceptance checks

- [ ] Checkpoint loads strictly into the preserved EfficientNet-B0 architecture.
- [ ] Checkpoint has exactly eight output logits.
- [ ] The class order is verified against training artifacts (not inferred from the number of outputs).
- [ ] Webcam face detection works.
- [ ] Model errors show “Emotion unavailable”; they never generate fake probabilities.
- [ ] Quit key reliably releases camera.

## Privacy and scope

- Camera frames are processed locally and not saved by this app.
- No identity recognition or enrollment UI is active yet. The identity registry scaffold requires explicit consent and stores only basic profile metadata; face templates are reserved for a later stage.
- Do not collect date of birth, blood type, or email unless a specific feature genuinely needs it. An expression classifier is not a mental-health assessment. MindPal should ask the person how they feel rather than infer their inner state from facial appearance.
- No API credentials are included. Rotate any credentials that appeared in previously uploaded source files.

=======
# MindPal
>>>>>>> 038d2efbfecd992974c3973e212a6c51f5c88e5d
