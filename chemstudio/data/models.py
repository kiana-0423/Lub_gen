from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Molecule(Base, TimestampMixin):
    __tablename__ = "molecules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    canonical_smiles: Mapped[str] = mapped_column(String(1024), unique=True, index=True)
    inchi: Mapped[str] = mapped_column(Text, default="")
    inchikey: Mapped[str] = mapped_column(String(64), default="", index=True)
    molblock: Mapped[str] = mapped_column(Text, default="")

    parameters: Mapped[list["MoleculeParameter"]] = relationship(back_populates="molecule", cascade="all, delete-orphan")
    descriptor_record: Mapped["MoleculeDescriptor | None"] = relationship(back_populates="molecule", uselist=False, cascade="all, delete-orphan")

    @property
    def display_name(self) -> str:
        return self.name or self.canonical_smiles


class MoleculeParameter(Base, TimestampMixin):
    __tablename__ = "molecule_parameters"
    __table_args__ = (UniqueConstraint("molecule_id", "key", name="uq_molecule_parameter_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    molecule_id: Mapped[int] = mapped_column(ForeignKey("molecules.id"), index=True)
    key: Mapped[str] = mapped_column(String(128))
    value: Mapped[str] = mapped_column(String(255))

    molecule: Mapped[Molecule] = relationship(back_populates="parameters")


class MoleculeDescriptor(Base, TimestampMixin):
    __tablename__ = "molecule_descriptors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    molecule_id: Mapped[int] = mapped_column(ForeignKey("molecules.id"), unique=True, index=True)
    descriptor_values: Mapped[dict] = mapped_column(JSON, default=dict)
    fingerprint_bits: Mapped[str] = mapped_column(Text, default="")

    molecule: Mapped[Molecule] = relationship(back_populates="descriptor_record")


class Formulation(Base, TimestampMixin):
    __tablename__ = "formulations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    notes: Mapped[str] = mapped_column(Text, default="")

    components: Mapped[list["FormulationComponent"]] = relationship(back_populates="formulation", cascade="all, delete-orphan")


class FormulationComponent(Base, TimestampMixin):
    __tablename__ = "formulation_components"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    formulation_id: Mapped[int] = mapped_column(ForeignKey("formulations.id"), index=True)
    molecule_id: Mapped[int] = mapped_column(ForeignKey("molecules.id"), index=True)
    ratio: Mapped[float] = mapped_column(Float, default=0.0)
    unit: Mapped[str] = mapped_column(String(64), default="wt%")

    formulation: Mapped[Formulation] = relationship(back_populates="components")


class TestCondition(Base, TimestampMixin):
    __tablename__ = "test_conditions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    temperature_c: Mapped[float] = mapped_column(Float, default=25.0)
    pressure_kpa: Mapped[float] = mapped_column(Float, default=101.3)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class Experiment(Base, TimestampMixin):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    formulation_id: Mapped[int] = mapped_column(ForeignKey("formulations.id"), index=True)
    test_condition_id: Mapped[int] = mapped_column(ForeignKey("test_conditions.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(64), default="draft")


class ExperimentResult(Base, TimestampMixin):
    __tablename__ = "experiment_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), index=True)
    metric_name: Mapped[str] = mapped_column(String(128))
    metric_value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(64), default="")


class ModelRecord(Base, TimestampMixin):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    model_type: Mapped[str] = mapped_column(String(128))
    target_name: Mapped[str] = mapped_column(String(128))
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_path: Mapped[str] = mapped_column(String(1024), default="")


class Prediction(Base, TimestampMixin):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), index=True)
    molecule_id: Mapped[int] = mapped_column(ForeignKey("molecules.id"), index=True)
    predicted_value: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)

