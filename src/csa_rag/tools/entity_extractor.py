from __future__ import annotations
from typing import List

class EntityExtractor:
    def __init__(self, model_name: str = "", labels: list[str] | None = None):
        self.labels = labels or ["person", "organization", "location", "date", "position", "event", "work"]
        self.model = None
        self.enabled = False

        if model_name:
            try:
                from gliner import GLiNER
                self.model = GLiNER.from_pretrained(model_name, local_files_only=True)
                self.enabled = True
                print(f"[EntityExtractor] GLiNER loaded from {model_name}")
            except Exception as e:
                print(f"[EntityExtractor] GLiNER disabled due to load error: {e}")
                self.model = None
                self.enabled = False

    def extract(self, text: str, threshold: float = 0.35) -> List[dict]:
        if not self.enabled or self.model is None:
            return []
        try:
            return self.model.predict_entities(text, self.labels, threshold=threshold)
        except Exception as e:
            print(f"[EntityExtractor] predict failed: {e}")
            return []
