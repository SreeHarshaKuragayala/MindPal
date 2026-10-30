"""MindPal desktop UI: local webcam, smoothed visual cues, and user chat."""
import html
import json
import os
import re
import sys
import time
import threading
from collections import deque

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt, QTimer, QRectF, QPointF
from openai import OpenAI
from dotenv import load_dotenv
from PySide6.QtGui import (
    QBrush, QColor, QPainter, QPen, QRadialGradient
)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget
)

from mindpal.emotion_model import EmotionModel
from mindpal.context_bridge import VisualContextBridge

HISTORY_SIZE = 8
INFERENCE_INTERVAL = 0.12
MIN_DISPLAY_SCORE = 0.45

# Aura colour per expression (R, G, B). The avatar glides between these.
AURA_COLORS = {
    "neutral":   (116, 148, 235),
    "happy":     (255, 183, 112),
    "sad":       (88, 118, 204),
    "angry":     (238, 112, 104),
    "fearful":   (158, 128, 232),
    "surprised": (104, 214, 226),
    "disgusted": (148, 196, 118),
    "contempt":  (190, 140, 172),
}
INK = (14, 18, 38)
WHITE = (255, 255, 255)


def mix(a, b, t):
    """Blend two RGB tuples; t=0 gives a, t=1 gives b."""
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def qcolor(rgb, alpha=255):
    return QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(alpha))


class CameraWorker(QThread):
    visual_status = Signal(str, str)
    error = Signal(str)

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self._running = True
        self.history = deque(maxlen=HISTORY_SIZE)
        self.last_inference = 0.0
        self.last_result = None

    def stop(self):
        self._running = False
        self.wait(3000)

    def run(self):
        cap = None
        try:
            model = EmotionModel(
                "models/emotion_model_pytorch.pth",
                "config/class_mapping.json",
            )
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(cascade_path)
            if detector.empty():
                raise RuntimeError("Could not load OpenCV Haar face detector.")

            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                raise RuntimeError(
                    f"Could not open camera {self.camera_index}. "
                    "Check camera permissions or try another camera index."
                )

            while self._running:
                ok, frame = cap.read()
                if not ok:
                    self.error.emit("Camera stopped returning frames.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=5, minSize=(70, 70)
                )

                if len(faces) == 0:
                    self.history.clear()
                    self.last_result = None
                    self.visual_status.emit(
                        "No face detected", "Visual context unavailable"
                    )
                else:
                    x, y, w, h = max(
                        faces, key=lambda b: int(b[2]) * int(b[3])
                    )
                    now = time.monotonic()
                    if (
                        self.last_result is None
                        or now - self.last_inference >= INFERENCE_INTERVAL
                    ):
                        try:
                            result = model.predict(frame[y:y+h, x:x+w])
                            scores = np.array(
                                [result["probabilities"][n] for n in model.class_names],
                                dtype=np.float64,
                            )
                            self.history.append(scores)
                            self.last_result = result
                            self.last_inference = now
                        except Exception as exc:
                            self.history.clear()
                            self.last_result = None
                            self.error.emit(f"Inference error: {exc}")

                    if self.history:
                        smoothed = np.mean(np.stack(self.history), axis=0)
                        idx = int(np.argmax(smoothed))
                        score = float(smoothed[idx])
                        cue = model.class_names[idx]
                        if score >= MIN_DISPLAY_SCORE:
                            text = f"Possible expression cue: {cue} ({score:.0%})"
                            self.visual_status.emit(text, "Tentative visual cue")
                            box_color = (50, 180, 110)
                        else:
                            self.visual_status.emit(
                                "Expression unclear", "Visual context uncertain"
                            )
                            box_color = (0, 165, 255)
                    else:
                        box_color = (0, 165, 255)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()


class AnimatedAvatar(QWidget):
    """Code-drawn companion: a softly breathing orb whose glow follows the
    detected expression, ripples while listening and pulses while speaking."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.expression = "neutral"
        self.speaking = False
        self.listening = False
        self.phase = 0.0
        self.breath = 0.0
        self.blink = False
        self.aura = AURA_COLORS["neutral"]
        self.aura_target = AURA_COLORS["neutral"]
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self.do_blink)
        self.blink_timer.start(3200)

    def set_expression(self, expression):
        allowed = set(AURA_COLORS)
        self.expression = expression if expression in allowed else "neutral"
        self.aura_target = AURA_COLORS[self.expression]
        self.update()

    def set_speaking(self, speaking):
        self.speaking = bool(speaking)
        if speaking:
            self.listening = False
        self.update()

    def set_listening(self, listening):
        self.listening = bool(listening)
        if listening:
            self.speaking = False
        self.update()

    def tick(self):
        self.phase += 0.16 + (0.35 if self.speaking else 0.0)
        self.breath += 0.045
        self.aura = mix(self.aura, self.aura_target, 0.07)
        self.update()

    def do_blink(self):
        self.blink = True
        QTimer.singleShot(150, self.finish_blink)

    def finish_blink(self):
        self.blink = False
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r = min(w * 0.28, h * 0.36) * (1 + 0.02 * np.sin(self.breath))
        c = QPointF(cx, cy)
        p.setPen(Qt.PenStyle.NoPen)

        # Soft ambient glow behind the face.
        glow = QRadialGradient(c, r * 2.2)
        glow.setColorAt(0.30, qcolor(self.aura, 95))
        glow.setColorAt(0.65, qcolor(self.aura, 28))
        glow.setColorAt(1.00, qcolor(self.aura, 0))
        p.setBrush(QBrush(glow))
        p.drawEllipse(c, r * 2.2, r * 2.2)

        # Listening: three ripples drift outward and fade.
        if self.listening:
            p.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(3):
                t = (self.phase * 0.05 + i / 3) % 1.0
                rr = r * (1.08 + 0.55 * t)
                p.setPen(QPen(qcolor(mix(self.aura, WHITE, 0.35), 150 * (1 - t)), 2))
                p.drawEllipse(c, rr, rr)
            p.setPen(Qt.PenStyle.NoPen)

        # Speaking: a steady ring that swells with the voice.
        if self.speaking:
            p.setBrush(Qt.BrushStyle.NoBrush)
            amp = abs(np.sin(self.phase))
            for k, a in ((0.0, 170), (0.14, 70)):
                rr = r * (1.07 + k + 0.05 * amp)
                p.setPen(QPen(qcolor(mix(self.aura, WHITE, 0.4), a), 3))
                p.drawEllipse(c, rr, rr)
            p.setPen(Qt.PenStyle.NoPen)

        # Face disc: lit from the upper left, tinted by the current aura.
        light = mix(mix(self.aura, WHITE, 0.10), INK, 0.15)
        dark = mix(self.aura, INK, 0.62)
        face = QRadialGradient(QPointF(cx - r * 0.35, cy - r * 0.45), r * 1.6)
        face.setColorAt(0.0, qcolor(light))
        face.setColorAt(1.0, qcolor(dark))
        p.setBrush(QBrush(face))
        p.drawEllipse(c, r, r)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 46), 2))
        p.drawEllipse(c, r - 1, r - 1)
        p.setPen(Qt.PenStyle.NoPen)

        # Eyes
        eye_y, eye_dx = cy - r * 0.16, r * 0.40
        eye_w, eye_h = r * 0.20, r * 0.17
        if self.expression == "surprised":
            eye_h *= 1.5
        elif self.expression in ("happy", "sad"):
            eye_h *= 0.75
        if self.blink:
            eye_h = 2
        drift = r * 0.018 * np.sin(self.breath * 0.7)
        for ex in (cx - eye_dx, cx + eye_dx):
            p.setBrush(QColor("#f4f7ff"))
            p.drawEllipse(QRectF(ex - eye_w / 2, eye_y - eye_h / 2, eye_w, max(2.0, eye_h)))
            if not self.blink:
                pupil = max(6.0, r * 0.085)
                p.setBrush(QColor("#151a33"))
                p.drawEllipse(QRectF(ex - pupil / 2 + drift, eye_y - pupil / 2, pupil, pupil))
                p.setBrush(QColor(255, 255, 255, 230))
                spark = pupil * 0.32
                p.drawEllipse(QRectF(ex + drift - pupil * 0.05, eye_y - pupil * 0.38, spark, spark))

        # Cheek blush when happy
        if self.expression == "happy":
            p.setBrush(QColor(255, 168, 150, 70))
            for sx in (-1, 1):
                p.drawEllipse(QRectF(cx + sx * r * 0.58 - r * 0.11, cy + r * 0.20, r * 0.22, r * 0.12))

        # Brows
        brow_y = eye_y - r * 0.25
        p.setPen(QPen(QColor(232, 240, 255, 200), max(2, r * 0.026), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        lx0, lx1 = cx - eye_dx - eye_w / 2, cx - eye_dx + eye_w / 2
        rx0, rx1 = cx + eye_dx - eye_w / 2, cx + eye_dx + eye_w / 2
        if self.expression == "sad":
            p.drawLine(QPointF(lx0, brow_y + 7), QPointF(lx1, brow_y - 2))
            p.drawLine(QPointF(rx0, brow_y - 2), QPointF(rx1, brow_y + 7))
        elif self.expression == "angry":
            p.drawLine(QPointF(lx0, brow_y - 3), QPointF(lx1, brow_y + 6))
            p.drawLine(QPointF(rx0, brow_y + 6), QPointF(rx1, brow_y - 3))
        else:
            lift = -5 if self.expression in ("surprised", "fearful") else 0
            p.drawLine(QPointF(lx0, brow_y + lift), QPointF(lx1, brow_y + lift))
            p.drawLine(QPointF(rx0, brow_y + lift), QPointF(rx1, brow_y + lift))

        # Mouth
        my = cy + r * 0.40
        p.setPen(QPen(QColor("#f4f7ff"), max(3, r * 0.036), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        if self.speaking:
            mh = r * (0.07 + 0.14 * abs(np.sin(self.phase)))
            p.setBrush(QColor("#151a33"))
            p.drawEllipse(QRectF(cx - r * 0.16, my - mh / 2, r * 0.32, max(5.0, mh)))
        elif self.expression == "happy":
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawArc(QRectF(cx - r * 0.24, my - r * 0.10, r * 0.48, r * 0.30), 205 * 16, 130 * 16)
        elif self.expression == "sad":
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawArc(QRectF(cx - r * 0.22, my - r * 0.01, r * 0.44, r * 0.25), 25 * 16, 130 * 16)
        elif self.expression == "surprised":
            p.setBrush(QColor("#151a33"))
            p.drawEllipse(QRectF(cx - r * 0.07, my - r * 0.09, r * 0.14, r * 0.20))
        else:
            p.drawLine(QPointF(cx - r * 0.16, my), QPointF(cx + r * 0.16, my))
        p.end()


class ChatWorker(QThread):
    """Runs the Nebius request off the UI thread."""
    answer_ready = Signal(str)
    request_failed = Signal(str)

    def __init__(self, messages, parent=None):
        super().__init__(parent)
        self.messages = messages

    def run(self):
        try:
            load_dotenv()
            api_key = os.getenv("NEBIUS_API_KEY")
            model = os.getenv("NEBIUS_MODEL")
            base_url = os.getenv(
                "NEBIUS_BASE_URL",
                "https://api.tokenfactory.us-north1.nebius.com/v1/"
            )
            if not api_key:
                raise RuntimeError(
                    "NEBIUS_API_KEY is missing. Add it to the .env file "
                    "in the MindPal project folder."
                )
            if not model:
                raise RuntimeError(
                    "NEBIUS_MODEL is missing. Add the exact model ID to .env."
                )

            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=self.messages,
            )
            answer = response.choices[0].message.content
            if not answer:
                answer = "I received an empty response. Could you try again?"
            self.answer_ready.emit(answer.strip())
        except Exception as exc:
            self.request_failed.emit(
                f"Nebius request failed: {exc}"
            )


class VoiceInputWorker(QThread):
    transcript_ready = Signal(str)
    status = Signal(str)
    request_failed = Signal(str)

    def run(self):
        try:
            import sounddevice as sd
            import numpy as np
            from faster_whisper import WhisperModel
            if not hasattr(VoiceInputWorker, "model"):
                VoiceInputWorker.model = WhisperModel("tiny", device="cpu", compute_type="int8")
            self.status.emit("Listening… speak naturally")
            audio = sd.rec(int(4 * 16000), samplerate=16000, channels=1, dtype="float32", blocking=True)
            self.status.emit("Understanding…")
            segments, _ = VoiceInputWorker.model.transcribe(np.asarray(audio).reshape(-1), language="en", vad_filter=True)
            transcript = " ".join(s.text.strip() for s in segments).strip()
            self.transcript_ready.emit(transcript)
        except Exception as exc:
            self.request_failed.emit("Microphone/transcription issue: " + str(exc) + ". Check Windows microphone privacy settings and input device.")


class SpeechOutputWorker(QThread):
    speech_started = Signal()
    speech_failed = Signal(str)
    speech_finished = Signal()

    def __init__(self, text, rate=175, parent=None):
        super().__init__(parent)
        self.text = text
        self.rate = rate
        self._stop_requested = threading.Event()

    def stop(self):
        self._stop_requested.set()

    def run(self):
        engine = None
        try:
            # pyttsx3 uses Windows SAPI5; initialize COM on this worker thread.
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except ImportError:
                pythoncom = None

            import pyttsx3
            engine = pyttsx3.init(driverName="sapi5")
            engine.setProperty("rate", self.rate)
            self.speech_started.emit()
            engine.say(self.text)
            # A non-blocking driver loop lets the user interrupt speech safely
            # from the UI without calling the SAPI engine across threads.
            engine.startLoop(False)
            while engine.isBusy() and not self._stop_requested.is_set():
                engine.iterate()
                time.sleep(0.02)
            if self._stop_requested.is_set():
                engine.stop()
            engine.endLoop()
            self.speech_finished.emit()
        except Exception as exc:
            self.speech_failed.emit(
                "Spoken reply failed: " + str(exc) + ". Check that Windows has a speech voice installed."
            )
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
            try:
                if 'pythoncom' in locals() and pythoncom is not None:
                    pythoncom.CoUninitialize()
            except Exception:
                pass


class MindPalWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MindPal — Voice Companion")
        self.resize(760, 860)
        self.setMinimumSize(560, 660)
        self.camera = None
        self.chat_worker = None
        self.voice_worker = None
        self.speech_worker = None
        self.latest_visual_cue = "No visual cue yet"
        self.chat_history = []
        self.listening_enabled = False
        self.proactive_started = False
        self.last_checkin_cue = None
        self._build_ui()

    def _build_ui(self):
        root = QWidget(); root.setObjectName("root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(32, 26, 32, 28)
        layout.setSpacing(12)

        title = QLabel("MindPal"); title.setObjectName("title"); title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle = QLabel("Your voice-first companion"); subtitle.setObjectName("muted"); subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.avatar = AnimatedAvatar(); self.avatar.setMinimumSize(440, 400)

        self.status = QLabel("Waiting for your permission"); self.status.setObjectName("status"); self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Conversation card: shows the latest line from you or MindPal.
        card = QFrame(); card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 16, 24, 16)
        self.caption = QLabel("A private space to talk, listen, and check in.")
        self.caption.setObjectName("caption")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setWordWrap(True)
        card_layout.addWidget(self.caption)
        card.setMinimumHeight(104)

        self.interrupt_button = QPushButton("Interrupt MindPal")
        self.interrupt_button.setObjectName("interrupt")
        self.interrupt_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.interrupt_button.setEnabled(False)
        self.interrupt_button.clicked.connect(self.interrupt_speech)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.avatar, 1)
        layout.addWidget(self.status, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(card)
        layout.addWidget(self.interrupt_button, 0, Qt.AlignmentFlag.AlignHCenter)
        self.setCentralWidget(root)

        self.setStyleSheet("""
            QWidget#root {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #171c38, stop:0.55 #0f1329, stop:1 #090c1b);
                color: #eef0ff;
                font-family: 'Segoe UI Variable Display', 'Segoe UI';
                font-size: 11pt;
            }
            QLabel#title { font-size: 28pt; font-weight: 600; color: #f3f4ff; background: transparent; }
            QLabel#muted { color: #9aa3c9; font-size: 11pt; background: transparent; }
            QLabel#status {
                color: #d4dcff; font-size: 11pt; font-weight: 600;
                background: rgba(255, 255, 255, 18);
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 17px; padding: 7px 20px;
            }
            QFrame#card {
                background: rgba(255, 255, 255, 12);
                border: 1px solid rgba(255, 255, 255, 24);
                border-radius: 20px;
            }
            QLabel#caption { color: #e4e8ff; font-size: 12pt; background: transparent; }
            QPushButton#interrupt {
                background: rgba(255, 255, 255, 16); color: #eef0ff;
                border: 1px solid rgba(160, 175, 255, 90);
                border-radius: 18px; padding: 10px 26px; font-weight: 600;
            }
            QPushButton#interrupt:hover { background: rgba(160, 175, 255, 45); }
            QPushButton#interrupt:disabled {
                color: #5f688c; background: rgba(255, 255, 255, 6);
                border-color: rgba(255, 255, 255, 18);
            }
        """)

    def show_turn(self, speaker, text):
        """Show a line of conversation with a small speaker label."""
        color = "#8f9bd6" if speaker == "You" else "#ffcf9e"
        self.caption.setText(
            f'<span style="color:{color}; font-size:10pt;">{speaker}</span><br>'
            f'{html.escape(text)}'
        )

    def show_privacy_and_start(self):
        policy = QMessageBox(self)
        policy.setWindowTitle("MindPal privacy and consent")
        policy.setIcon(QMessageBox.Icon.Information)
        policy.setText("Before we begin, please review how MindPal works.")
        policy.setInformativeText(
            "• Camera: your camera is used locally on this computer to detect a face and estimate tentative expression cues. No camera images or video are sent to Nebius.\n"
            "• Microphone: short audio segments are transcribed locally on this computer.\n"
            "• AI conversation: your transcribed words and a brief conversation context are sent to Nebius Token Factory to generate replies.\n"
            "• Voice: replies are spoken aloud using Windows text-to-speech.\n"
            "• Limitations: expression estimates can be wrong; MindPal is not a medical or diagnostic service.\n\n"
            "Choose I Agree to start camera and microphone access. You can close MindPal at any time to stop both."
        )
        policy.setStandardButtons(QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok)
        policy.button(QMessageBox.StandardButton.Ok).setText("I Agree — Start MindPal")
        if policy.exec() == QMessageBox.StandardButton.Ok:
            self.start_session()
        else:
            self.status.setText("Not started — consent was not given")
            self.caption.setText("Close the app or restart it when you are ready.")

    def start_session(self):
        self.status.setText("Starting camera and microphone…")
        self.caption.setText("Your camera stays hidden. Talk naturally when MindPal is listening.")
        self.camera = CameraWorker(0, self)
        self.camera.visual_status.connect(self.on_visual_status)
        self.camera.error.connect(self.on_camera_error)
        self.camera.start()
        self.listening_enabled = True
        # Speak the welcome first; microphone listening begins after playback to avoid echo.
        QTimer.singleShot(1800, self.initial_greeting)

    def initial_greeting(self):
        if not self.listening_enabled: return
        self.ask_nebius("Greet me naturally like a familiar companion. Keep it brief and ask one open, low-pressure question to start our conversation. Do not mention camera or emotion detection.", proactive=True)

    def on_visual_status(self, cue, _status):
        self.latest_visual_cue = cue
        if cue.startswith("Possible expression cue:"):
            detected = cue.split(":",1)[1].split("(",1)[0].strip().lower()
            mapping = {"happiness":"happy", "sadness":"sad", "disgusted":"disgusted", "fearful":"fearful", "surprised":"surprised", "angry":"angry", "contempt":"contempt", "neutral":"neutral"}
            self.avatar.set_expression(mapping.get(detected, "neutral"))
            # Do not interrupt an active conversation. A changed cue is only
            # made available to the model for the next natural reply.
            self.last_checkin_cue = detected
        elif cue.startswith("No face") or cue.startswith("Expression unclear"):
            self.avatar.set_expression("neutral")

    def on_camera_error(self, message):
        self.status.setText("Camera unavailable — voice chat can continue")
        self.caption.setText(message)

    def start_listening(self):
        if not self.listening_enabled: return
        if (self.voice_worker and self.voice_worker.isRunning()) or (self.speech_worker and self.speech_worker.isRunning()) or (self.chat_worker and self.chat_worker.isRunning()):
            return
        self.voice_worker = VoiceInputWorker(self)
        self.voice_worker.status.connect(self.set_status)
        self.voice_worker.status.connect(self.on_voice_status)
        self.voice_worker.transcript_ready.connect(self.on_transcript)
        self.voice_worker.request_failed.connect(self.on_voice_error)
        self.voice_worker.finished.connect(lambda: QTimer.singleShot(250, self.start_listening) if self.listening_enabled and not (self.chat_worker and self.chat_worker.isRunning()) and not (self.speech_worker and self.speech_worker.isRunning()) else None)
        self.voice_worker.start()

    def set_status(self, text): self.status.setText(text)

    def on_voice_status(self, text):
        self.avatar.set_listening(text.startswith("Listening"))

    def on_voice_error(self, message):
        self.status.setText("Microphone needs attention")
        self.caption.setText(message)
        QTimer.singleShot(1800, self.start_listening)

    def on_transcript(self, text):
        if not text:
            self.status.setText("Listening… say something whenever you're ready")
            QTimer.singleShot(200, self.start_listening)
            return
        self.avatar.set_listening(False)
        self.show_turn("You", text)
        self.ask_nebius(text)

    def ask_nebius(self, user_text, proactive=False):
        if self.chat_worker and self.chat_worker.isRunning(): return
        system = (
            "You are MindPal, a warm, natural, voice-first companion. Have a real conversation, not a sequence of generic chatbot answers. "
            "Remember details already shared in this conversation, refer back to them naturally, and avoid asking the same question twice. "
            "Speak in short, human-sounding sentences that are easy to hear aloud. Use contractions and natural pauses, but no markdown, lists, headings, emojis, or stage directions. "
            "The user's spoken words and stated preferences are the primary source of context. Never diagnose or claim to know how the user feels. "
            "A local facial-expression estimate is uncertain and may be wrong; use it only as a very weak signal to adjust warmth, never mention the camera, the estimate, or label an emotion. "
            "If the user sounds distressed from their own words, slow down, acknowledge what they said, and ask at most one gentle question. Otherwise keep the conversation relaxed and curious. "
            "Do not repeatedly initiate check-ins or interrupt the user. If there is immediate danger or intent to self-harm, encourage contacting local emergency services or a trusted person."
        )
        messages = [{"role":"system", "content":system}]
        messages.append({"role":"system", "content":"Optional uncertain local visual cue: " + self.latest_visual_cue + ". Use only for subtle tone adaptation; do not mention it."})
        messages.extend(self.chat_history[-10:])
        if not proactive:
            messages.append({"role":"user", "content":user_text})
            self.chat_history.append({"role":"user", "content":user_text})
        self.status.setText("MindPal is thinking…")
        self.chat_worker = ChatWorker(messages, self)
        self.chat_worker.answer_ready.connect(self.on_answer)
        self.chat_worker.request_failed.connect(self.on_ai_error)
        self.chat_worker.start()

    def on_answer(self, answer):
        self.chat_history.append({"role":"assistant", "content":answer})
        self.show_turn("MindPal", answer)
        cue = self.latest_visual_cue.lower()
        slower_context = any(word in cue for word in ("sadness", "fearful")) or any(
            word in answer.lower() for word in ("take your time", "no rush", "that sounds difficult")
        )
        speaking_rate = 158 if slower_context else 175
        self.speech_worker = SpeechOutputWorker(answer, rate=speaking_rate, parent=self)
        self.speech_worker.speech_started.connect(self.on_speech_started)
        self.speech_worker.speech_finished.connect(self.on_speech_finished)
        self.speech_worker.speech_failed.connect(self.on_speech_error)
        self.speech_worker.start()

    def on_speech_started(self):
        self.avatar.set_speaking(True)
        self.interrupt_button.setEnabled(True)
        self.status.setText("MindPal is speaking…")

    def interrupt_speech(self):
        if self.speech_worker and self.speech_worker.isRunning():
            self.speech_worker.stop()
            self.status.setText("Okay — I'll listen.")

    def on_speech_finished(self):
        self.interrupt_button.setEnabled(False)
        self.avatar.set_speaking(False)
        self.avatar.set_listening(True)
        self.status.setText("Listening…")
        QTimer.singleShot(350, self.start_listening)

    def on_speech_error(self, message):
        self.interrupt_button.setEnabled(False)
        self.avatar.set_speaking(False)
        self.avatar.set_listening(False)
        self.status.setText("Speech output unavailable")
        self.caption.setText(message)
        QTimer.singleShot(800, self.start_listening)

    def on_ai_error(self, message):
        self.avatar.set_listening(False)
        self.status.setText("AI connection issue")
        self.caption.setText(message)
        QTimer.singleShot(1000, self.start_listening)

    def closeEvent(self, event):
        self.listening_enabled = False
        if self.camera and self.camera.isRunning(): self.camera.stop()
        if self.voice_worker and self.voice_worker.isRunning(): self.voice_worker.wait(2000)
        if self.chat_worker and self.chat_worker.isRunning(): self.chat_worker.wait(5000)
        if self.speech_worker and self.speech_worker.isRunning(): self.speech_worker.wait(3000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MindPalWindow()
    window.show()
    QTimer.singleShot(300, window.show_privacy_and_start)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()