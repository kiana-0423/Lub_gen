from __future__ import annotations

try:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:  # pragma: no cover
    Chem = None
    rdFingerprintGenerator = None


class MorganFingerprintFeaturizer:
    def featurize(self, smiles: str) -> list[int]:
        if Chem is None:
            raise ValueError("RDKit is not installed.")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES.")
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
        fp = generator.GetFingerprint(mol)
        return list(fp)
