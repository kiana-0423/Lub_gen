from __future__ import annotations

from chemstudio.workers._qt_compat import QThread, Signal


class TrainWorker(QThread):
    result_ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, model_service, training_kwargs: dict) -> None:
        super().__init__()
        self.model_service = model_service
        self.training_kwargs = training_kwargs

    def run(self) -> None:
        try:
            result = self.model_service.train_model(**self.training_kwargs)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))
            return
        self.result_ready.emit(result)
