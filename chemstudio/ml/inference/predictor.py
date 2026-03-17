from __future__ import annotations


class Predictor:
    def predict(self, features: dict) -> dict:
        return {"value": 0.0, "confidence": 0.0, "features": features}

