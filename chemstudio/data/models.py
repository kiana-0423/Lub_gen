from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(UTC)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Molecule(Base, TimestampMixin):
    __tablename__ = "molecules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str | None] = mapped_column(String(64), unique=True, index=True, nullable=True)
    name: Mapped[str] = mapped_column(String(255))
    input_smiles: Mapped[str] = mapped_column(String(1024), default="")
    canonical_smiles: Mapped[str] = mapped_column(String(1024), unique=True, index=True)
    inchi: Mapped[str] = mapped_column(Text, default="")
    inchikey: Mapped[str] = mapped_column(String(64), default="", index=True)
    molblock: Mapped[str] = mapped_column(Text, default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    is_hidden: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    parameters: Mapped[list["MoleculeParameter"]] = relationship(
        back_populates="molecule",
        cascade="all, delete-orphan",
    )
    descriptor_record: Mapped["MoleculeDescriptor | None"] = relationship(
        back_populates="molecule",
        uselist=False,
        cascade="all, delete-orphan",
    )
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="molecule")

    @property
    def display_name(self) -> str:
        return self.name or self.code or self.canonical_smiles


class MoleculeParameter(Base, TimestampMixin):
    __tablename__ = "molecule_parameters"
    __table_args__ = (UniqueConstraint("molecule_id", "key", name="uq_molecule_parameter_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    molecule_id: Mapped[int] = mapped_column(ForeignKey("molecules.id"), index=True)
    key: Mapped[str] = mapped_column(String(128))
    value: Mapped[str] = mapped_column(String(1024))

    molecule: Mapped[Molecule] = relationship(back_populates="parameters")


class MoleculeDescriptor(Base, TimestampMixin):
    __tablename__ = "molecule_descriptors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    molecule_id: Mapped[int] = mapped_column(ForeignKey("molecules.id"), unique=True, index=True)
    descriptor_values: Mapped[dict] = mapped_column(JSON, default=dict)
    fingerprint_bits: Mapped[str] = mapped_column(Text, default="")

    molecule: Mapped[Molecule] = relationship(back_populates="descriptor_record")


class ModelRecord(Base, TimestampMixin):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    model_type: Mapped[str] = mapped_column(String(128))
    problem_type: Mapped[str] = mapped_column(String(64))
    target_name: Mapped[str] = mapped_column(String(128))
    feature_columns_json: Mapped[list] = mapped_column(JSON, default=list)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    training_config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_path: Mapped[str] = mapped_column(String(1024), default="")

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model")


class Prediction(Base, TimestampMixin):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), index=True)
    molecule_id: Mapped[int | None] = mapped_column(ForeignKey("molecules.id"), index=True, nullable=True)
    predicted_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_label: Mapped[str] = mapped_column(String(255), default="")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_features_json: Mapped[dict] = mapped_column(JSON, default=dict)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)

    model: Mapped[ModelRecord] = relationship(back_populates="predictions")
    molecule: Mapped[Molecule | None] = relationship(back_populates="predictions")
