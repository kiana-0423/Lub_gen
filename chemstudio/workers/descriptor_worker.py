from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class DescriptorWorker(QThread):
    result_ready = Signal(dict)
    failed = Signal(str)

    def __init__(self, descriptor_service, smiles: str, molecule_id: int | None = None) -> None:
        super().__init__()
        self.descriptor_service = descriptor_service
        self.smiles = smiles
        self.molecule_id = molecule_id

    def run(self) -> None:
        try:
            if self.molecule_id is not None:
                descriptors = self.descriptor_service.calculate_and_persist(self.molecule_id, self.smiles)
            else:
                descriptors = self.descriptor_service.calculate(self.smiles)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))
            return
        self.result_ready.emit(descriptors)
