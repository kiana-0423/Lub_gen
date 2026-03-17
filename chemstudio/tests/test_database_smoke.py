from __future__ import annotations

from chemstudio.data.db import initialize_database, session_scope
from chemstudio.data.models import Molecule


def test_database_initializes():
    initialize_database()
    with session_scope() as session:
        assert session.query(Molecule).count() >= 0

