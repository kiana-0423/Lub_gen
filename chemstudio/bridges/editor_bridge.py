from __future__ import annotations

import json

from PySide6.QtCore import QObject, Signal, Slot


class EditorBridge(QObject):
    structureChanged = Signal(dict)
    moleculeSaved = Signal(dict)
    descriptorsReady = Signal(dict)
    logMessage = Signal(str)

    def __init__(self, molecule_service, descriptor_service) -> None:
        super().__init__()
        self.molecule_service = molecule_service
        self.descriptor_service = descriptor_service
        self._payload: dict = {"id": None, "name": "", "smiles": "", "molblock": ""}

    @Slot(str)
    def onStructureEdited(self, payload_json: str) -> None:
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            self.logMessage.emit("Ignoring invalid payload from web editor.")
            return

        self._payload.update(payload)
        self.structureChanged.emit(self._payload.copy())

    def update_payload(self, payload: dict) -> None:
        self._payload.update(payload)
        self.structureChanged.emit(self._payload.copy())

    def get_current_payload(self) -> dict:
        return self._payload.copy()

    def emit_molecule_saved(self, payload: dict) -> None:
        self.moleculeSaved.emit(payload)

    def emit_descriptors(self, descriptors: dict) -> None:
        self.descriptorsReady.emit(descriptors)

