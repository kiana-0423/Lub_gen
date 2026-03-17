from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QLabel, QLineEdit, QPlainTextEdit, QVBoxLayout, QWidget

try:
    from PySide6.QtWebChannel import QWebChannel
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # pragma: no cover
    QWebChannel = None
    QWebEngineView = None


class MoleculeEditorWidget(QWidget):
    def __init__(self, editor_bridge) -> None:
        super().__init__()
        self.editor_bridge = editor_bridge
        self._payload: dict = {"id": None, "name": "", "smiles": "", "molblock": ""}
        self._web_view = None
        self._name_input = None
        self._smiles_input = None
        self._molblock_input = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if self._should_use_plain_editor():
            self._build_plain_editor(layout)
            self.editor_bridge.logMessage.emit("Using plain Qt editor fallback.")
        else:
            self._build_web_editor(layout)

    def _should_use_plain_editor(self) -> bool:
        requested = os.environ.get("CHEMSTUDIO_DISABLE_WEBENGINE", "").strip() == "1"
        unavailable = QWebEngineView is None or QWebChannel is None
        return requested or unavailable

    def _build_web_editor(self, layout: QVBoxLayout) -> None:
        self._web_view = QWebEngineView(self)
        channel = QWebChannel(self._web_view.page())
        channel.registerObject("editorBridge", self.editor_bridge)
        self._web_view.page().setWebChannel(channel)
        self._channel = channel
        layout.addWidget(self._web_view)
        self._load_editor()

    def _build_plain_editor(self, layout: QVBoxLayout) -> None:
        layout.addWidget(QLabel("WebEngine disabled. Using plain editor fallback.", self))

        self._name_input = QLineEdit(self)
        self._name_input.setPlaceholderText("Molecule Name")
        self._smiles_input = QLineEdit(self)
        self._smiles_input.setPlaceholderText("SMILES")
        self._molblock_input = QPlainTextEdit(self)
        self._molblock_input.setPlaceholderText("MolBlock / notes")

        self._name_input.textChanged.connect(self._sync_plain_editor)
        self._smiles_input.textChanged.connect(self._sync_plain_editor)
        self._molblock_input.textChanged.connect(self._sync_plain_editor)

        layout.addWidget(self._name_input)
        layout.addWidget(self._smiles_input)
        layout.addWidget(self._molblock_input)
        self._sync_plain_editor()

    def _load_editor(self) -> None:
        html_path = Path(__file__).resolve().parents[1] / "web" / "index.html"
        self._web_view.load(QUrl.fromLocalFile(str(html_path)))

    def request_structure_payload(self) -> dict:
        if self._web_view is None:
            return self._payload.copy()
        return self.editor_bridge.get_current_payload()

    def load_structure(self, payload: dict) -> None:
        self._payload.update(payload)
        if self._web_view is not None:
            payload_json = json.dumps(payload)
            script = f"window.chemstudioEditor && window.chemstudioEditor.loadStructure({payload_json});"
            self._web_view.page().runJavaScript(script)
        else:
            self._name_input.setText(payload.get("name") or "")
            self._smiles_input.setText(payload.get("smiles") or "")
            self._molblock_input.setPlainText(payload.get("molblock") or "")
        self.editor_bridge.update_payload(payload)

    def _sync_plain_editor(self) -> None:
        self._payload.update(
            {
                "name": self._name_input.text().strip(),
                "smiles": self._smiles_input.text().strip(),
                "molblock": self._molblock_input.toPlainText(),
            }
        )
        self.editor_bridge.update_payload(self._payload)
