from __future__ import annotations

from chemstudio.services.model_service import ModelService


class RegressionTrainer:
    def __init__(self) -> None:
        self.service = ModelService()

