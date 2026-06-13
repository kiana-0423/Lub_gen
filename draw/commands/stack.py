"""Command stack wrapper for future editor actions."""

from __future__ import annotations

from PySide6.QtGui import QAction, QUndoCommand, QUndoStack


class EditorCommandStack:
    """Thin wrapper around ``QUndoStack`` to isolate command infrastructure."""

    def __init__(self) -> None:
        self._stack = QUndoStack()

    @property
    def stack(self) -> QUndoStack:
        """Expose the underlying Qt undo stack."""
        return self._stack

    def create_undo_action(self, parent: object, text: str = "Undo") -> QAction:
        """Create a standard undo action."""
        return self._stack.createUndoAction(parent, text)

    def create_redo_action(self, parent: object, text: str = "Redo") -> QAction:
        """Create a standard redo action."""
        return self._stack.createRedoAction(parent, text)

    def push(self, command: QUndoCommand) -> None:
        """Push a new undoable command."""
        self._stack.push(command)

    def undo(self) -> None:
        """Undo the latest command."""
        self._stack.undo()

    def redo(self) -> None:
        """Redo the latest command."""
        self._stack.redo()

    def clear(self) -> None:
        """Clear the stack."""
        self._stack.clear()
