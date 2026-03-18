from __future__ import annotations

try:
    from PySide6.QtCore import QThread, Signal
except ImportError:  # pragma: no cover
    class Signal:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            self._subscribers: list = []

        def connect(self, callback) -> None:
            self._subscribers.append(callback)

        def emit(self, *args, **kwargs) -> None:
            for callback in list(self._subscribers):
                callback(*args, **kwargs)

    class QThread:  # pragma: no cover
        def start(self) -> None:
            self.run()

        def run(self) -> None:
            return None
