
"""MindPal Stage 1: live vision, temporal smoothing, and user reports."""

import argparse
import logging
import time
import threading
from collections import deque
from queue import Queue, Empty

import cv2
import numpy as np

from mindpal.emotion_model import EmotionModel
from mindpal.context_bridge import VisualContextBridge

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

INFERENCE_INTERVAL = 0.12
HISTORY_SIZE = 8
MIN_DISPLAY_SCORE = 0.45


def terminal_input(report_queue):
    """Collect user reports without blocking the webcam loop."""
    print("\nYou can type how you feel at any time.")
    print("Type a report and press Enter; type /quit to exit MindPal.\n")

    while True:
        try:
            report = input("How are you feeling? > ").strip()
        except (EOFError, KeyboardInterrupt):
            report_queue.put("/quit")
            return

        if report:
            report_queue.put(report)

        if report.lower() == "/quit":
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/emotion_model_pytorch.pth")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mapping", default="config/class_mapping.json")
    args = parser.parse_args()

    emotion_model = EmotionModel(args.weights, args.mapping)
    bridge = VisualContextBridge(min_score=MIN_DISPLAY_SCORE)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try --camera 1.")

    report_queue = Queue()
    input_thread = threading.Thread(
        target=terminal_input,
        args=(report_queue,),
        daemon=True,
    )
    input_thread.start()

    history = deque(maxlen=HISTORY_SIZE)
    last_inference_time = 0.0
    last_result = None
    last_face_box = None
    user_report = None
    should_quit = False

    print("\nMindPal running on the camera loop.")
    print("Visual estimates are uncertain cues, not proof of feelings.")
    print("Press Q or Esc in the camera window to exit.\n")

    try:
        while not should_quit:
            # Process reports from the terminal without blocking camera frames.
            while True:
                try:
                    report = report_queue.get_nowait()
                except Empty:
                    break

                if report.lower() == "/quit":
                    should_quit = True
                    break

                user_report = report
                print(f"\nYour report saved for this session: {user_report}")

                current_probs = (
                    {
                        name: float(score)
                        for name, score in zip(
                            emotion_model.class_names,
                            np.mean(np.stack(history), axis=0),
                        )
                    }
                    if history else None
                )
                context = bridge.build_context(current_probs, user_report)
                print("Context status:", context["status"])
                print("User report:", context["user_report"])
                print("Visual cue:", context["visual_cue"])
                print("Suggested approach:", context["prompt_suggestion"], "\n")

            if should_quit:
                break

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

            label = "No face detected"
            color = (0, 165, 255)
            last_face_box = None

            if len(faces) == 0:
                history.clear()
                last_result = None
            else:
                # Stage 1 follows the largest visible face.
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
                        last_result = emotion_model.predict(crop)
                        current = np.array(
                            [
                                last_result["probabilities"][name]
                                for name in emotion_model.class_names
                            ],
                            dtype=np.float64,
                        )
                        history.append(current)
                        last_inference_time = now
                    except Exception:
                        logging.exception("Live inference failed.")
                        history.clear()
                        last_result = None

                if history:
                    smoothed = np.mean(np.stack(history), axis=0)
                    best_index = int(np.argmax(smoothed))
                    score = float(smoothed[best_index])
                    emotion = emotion_model.class_names[best_index]

                    if score >= MIN_DISPLAY_SCORE:
                        label = f"Visual cue: {emotion} ({score:.0%})"
                        color = (40, 190, 40)
                    else:
                        label = "Expression unclear"

                # Build context without treating the model output as fact.
                current_probs = (
                    {
                        name: float(score)
                        for name, score in zip(
                            emotion_model.class_names,
                            np.mean(np.stack(history), axis=0),
                        )
                    }
                    if history else None
                )
                context = bridge.build_context(current_probs, user_report)

            if last_face_box is not None:
                x, y, w, h = last_face_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame, label, (x, max(24, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
                )

            cv2.putText(
                frame, "MindPal | Q/Esc to quit", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, "Visual cues are uncertain - ask, don't assume",
                (12, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow("MindPal - Stage 1", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nMindPal camera loop stopped.")


if __name__ == "__main__":
    main()
