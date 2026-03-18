from __future__ import annotations

from chemstudio.data.db import initialize_database, session_scope
from chemstudio.data.models import ModelRecord, Molecule, MoleculeDescriptor, MoleculeParameter, Prediction


def test_database_initializes(chemstudio_env):
    initialize_database()
    with session_scope() as session:
        assert session.query(Molecule).count() == 0
        assert session.query(MoleculeParameter).count() == 0
        assert session.query(MoleculeDescriptor).count() == 0
        assert session.query(ModelRecord).count() == 0
        assert session.query(Prediction).count() == 0
