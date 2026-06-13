"""Editor tool definitions."""

from __future__ import annotations

from enum import StrEnum


class EditorTool(StrEnum):
    """High-level tools supported by the molecule editor MVP."""

    SELECT = "Select"
    ATOM = "Atom"
    BOND = "Bond"
    DELETE = "Delete"


ALL_TOOLS = tuple(tool.value for tool in EditorTool)

