from __future__ import annotations

from chemstudio.workers._qt_compat import QThread, Signal


class ImportWorker(QThread):
    result_ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, molecule_service, records) -> None:
        super().__init__()
        self.molecule_service = molecule_service
        self.records = records

    def run(self) -> None:
        try:
            result = self.molecule_service.import_molecules(self.records)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))
            return
        self.result_ready.emit(result)
