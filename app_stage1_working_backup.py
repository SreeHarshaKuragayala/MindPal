
"""MindPal Stage 1: webcam, CUDA inference, and temporal smoothing."""

import argparse
import logging
import time
from collections import deque

import cv2
import numpy as np

from mindpal.emotion_model import EmotionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

INFERENCE_INTERVAL = 0.12  # About 8 predictions per second
HISTORY_SIZE = 8           # Up to about 1 second of predictions
MIN_DISPLAY_SCORE = 0.45   # Heuristic, not a calibrated probability threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", default="models/emotion_model_pytorch.pth"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--mapping", default="config/class_mapping.json"
    )
    args = parser.parse_args()

    emotion_model = EmotionModel(args.weights, args.mapping)

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError(
            f"Could not load Haar cascade: {cascade_path}"
        )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Try --camera 1."
        )

    # Each entry contains the model's probability distribution
    # for a recent frame of the primary face.
    history = deque(maxlen=HISTORY_SIZE)
    last_inference_time = 0.0
    last_result = None
    last_face_box = None

    print("MindPal Stage 1 running. Press Q or ESC to quit.")
    print(
        "Smoothed expression estimates are uncertain visual cues, "
        "not proof of how someone feels."
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logging.warning("No frame received from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(70, 70),
            )

            if len(faces) == 0:
                history.clear()
                last_result = None
                last_face_box = None
                label = "No face detected"
            else:
                # Focus on the largest face for this single-user stage.
                x, y, w, h = max(
                    faces, key=lambda box: int(box[2]) * int(box[3])
                )
                last_face_box = (x, y, w, h)

                now = time.monotonic()
                if (
                    last_result is None
                    or now - last_inference_time >= INFERENCE_INTERVAL
                ):
                    crop = frame[y:y + h, x:x + w]
                    try:
                        result = emotion_model.predict(crop)
                        probabilities = result["probabilities"]

                        # Keep the class order from the verified model mapping.
                        current = np.array(
                            [
                                probabilities[name]
                                for name in emotion_model.class_names
                            ],
                            dtype=np.float64,
                        )

                        history.append(current)
                        last_result = result
                        last_inference_time = now

                    except Exception:
                        logging.exception("Live inference failed.")
                        history.clear()
                        last_result = None

                if history and last_result is not None:
                    # Average recent probability vectors before selecting
                    # the displayed class.
                    smoothed = np.mean(
                        np.stack(history, axis=0), axis=0
                    )
                    best_index = int(np.argmax(smoothed))
                    score = float(smoothed[best_index])
                    emotion = emotion_model.class_names[best_index]

                    if score < MIN_DISPLAY_SCORE:
                        label = "Expression unclear"
                        color = (0, 165, 255)
                    else:
                        label = f"Estimate: {emotion} ({score:.0%})"
                        color = (40, 190, 40)
                else:
                    label = "Expression unavailable"
                    color = (0, 165, 255)

            if last_face_box is not None:
                x, y, w, h = last_face_box
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), color, 2
                )
                cv2.putText(
                    frame,
                    label,
                    (x, max(24, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                "MindPal | Q/Esc to quit",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Expression estimate only - ask how the person feels",
                (12, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("MindPal - Stage 1", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
