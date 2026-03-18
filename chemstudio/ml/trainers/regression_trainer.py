from __future__ import annotations

from chemstudio.services.model_service import ModelService


class RegressionTrainer:
    def __init__(self, model_service: ModelService | None = None) -> None:
        self.model_service = model_service or ModelService()

    def train(self, **kwargs):
        return self.model_service.train_model(problem_type="regression", **kwargs)
