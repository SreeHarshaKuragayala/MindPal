"""Small local identity registry scaffold. No face images or biometrics are stored yet."""
import json
from pathlib import Path
from datetime import datetime, timezone

class IdentityStore:
    def __init__(self, path="data/identities.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text('{"identities": []}\n', encoding="utf-8")

    def list_profiles(self):
        return json.loads(self.path.read_text(encoding="utf-8")).get("identities", [])

    def add_profile(self, *, display_name: str, consent: bool, email: str | None = None):
        """Enrollment hook for later stages; requires explicit consent."""
        name = display_name.strip()
        if not consent:
            raise PermissionError("Identity enrollment requires explicit consent.")
        if not name:
            raise ValueError("A display name is required.")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        profile = {
            "id": f"local-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "display_name": name,
            "email": email.strip() if email else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "consent_recorded": True,
            "face_templates": [],
        }
        data.setdefault("identities", []).append(profile)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return profile
