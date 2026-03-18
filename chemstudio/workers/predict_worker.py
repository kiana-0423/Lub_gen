from __future__ import annotations

from chemstudio.workers._qt_compat import QThread, Signal


class PredictWorker(QThread):
    result_ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, model_service, prediction_kwargs: dict, *, single: bool = False) -> None:
        super().__init__()
        self.model_service = model_service
        self.prediction_kwargs = prediction_kwargs
        self.single = single

    def run(self) -> None:
        try:
            if self.single:
                result = self.model_service.predict_single(**self.prediction_kwargs)
            else:
                result = self.model_service.predict_for_molecules(**self.prediction_kwargs)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))
            return
        self.result_ready.emit(result)
