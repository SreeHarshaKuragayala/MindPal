
"""MindPal: convert visual predictions into cautious conversation context.

Visual predictions are uncertain cues, not verified emotional states.
This module does not diagnose, identify, or store personal information.
"""


class VisualContextBridge:
    def __init__(self, min_score: float = 0.45):
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1.")
        self.min_score = min_score

    def build_context(
        self,
        probabilities: dict[str, float] | None,
        user_report: str | None = None,
    ) -> dict:
        """Build a safe context payload for a future conversation module."""

        report = user_report.strip() if user_report else None

        base = {
            "visual_cue": None,
            "visual_score": None,
            "user_report": report,
            "source": "visual_model",
            "prompt_suggestion": (
                "Ask the user how they are feeling, without making assumptions."
            ),
        }

        if not probabilities:
            base["status"] = "unavailable"
            return base

        # Reject malformed, negative, or non-finite scores.
        import math

        scores = {}
        for name, value in probabilities.items():
            try:
                score = float(value)
            except (TypeError, ValueError):
                base["status"] = "unavailable"
                return base

            if not math.isfinite(score) or score < 0.0:
                base["status"] = "unavailable"
                return base

            scores[str(name)] = score

        total = sum(scores.values())
        if not scores or total <= 0.0:
            base["status"] = "unavailable"
            return base

        # Normalize defensively in case the caller's scores do not sum to 1.
        normalized = {
            name: score / total for name, score in scores.items()
        }
        cue = max(normalized, key=normalized.get)
        score = normalized[cue]

        if score < self.min_score:
            base["status"] = "uncertain"
            return base

        base["visual_cue"] = cue
        base["visual_score"] = round(score, 4)
        base["status"] = "tentative"

        # The visual cue is never converted into a claim about the user's
        # actual feelings. The conversational system should ask, not assume.
        base["prompt_suggestion"] = (
            "You may gently ask how the user is feeling. "
            "Do not infer their feelings from their facial expression."
        )

        return base
