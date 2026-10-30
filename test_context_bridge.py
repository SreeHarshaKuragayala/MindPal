
from mindpal.context_bridge import VisualContextBridge

bridge = VisualContextBridge(min_score=0.45)

# A clear model output: still only a tentative visual cue.
result = bridge.build_context({
    "happiness": 0.82,
    "neutral": 0.10,
    "sadness": 0.08,
})
print("Clear cue:", result)

# Ambiguous output should not expose a winning label.
result = bridge.build_context({
    "happiness": 0.38,
    "neutral": 0.34,
    "sadness": 0.28,
})
print("Uncertain cue:", result)

# The user's own report is kept separate from visual inference.
result = bridge.build_context(
    {"sadness": 0.72, "neutral": 0.28},
    user_report="I'm feeling tired today.",
)
print("User-confirmed context:", result)

# Missing visual input must remain unavailable.
print("No input:", bridge.build_context(None))
