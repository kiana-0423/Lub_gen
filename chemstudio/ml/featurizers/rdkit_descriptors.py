from __future__ import annotations

from chemstudio.services.descriptor_service import DescriptorService


class RDKitDescriptorFeaturizer:
    def __init__(self) -> None:
        self.service = DescriptorService()

    def featurize(self, smiles: str) -> dict[str, float | int]:
        return self.service.calculate(smiles)
