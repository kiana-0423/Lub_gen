from __future__ import annotations

import os

from chemstudio.utils.config import AppConfig, ensure_runtime_directories
from chemstudio.utils.file_utils import ensure_directory

ensure_runtime_directories()
os.environ.setdefault("MPLCONFIGDIR", str(ensure_directory(AppConfig.LOG_DIR / "matplotlib")))

import matplotlib.axes
import pandas as pd


class VisualizationService:
    """Centralizes matplotlib plotting for the desktop UI."""

    def plot_distribution(self, axes: matplotlib.axes.Axes, dataframe: pd.DataFrame, column_name: str) -> None:
        """Plot a histogram for a numeric column."""
        axes.clear()
        series = pd.to_numeric(dataframe[column_name], errors="coerce").dropna()
        axes.hist(series, bins=min(12, max(5, len(series))), color="#2c7fb8", edgecolor="white")
        axes.set_title(f"{column_name} distribution")
        axes.set_xlabel(column_name)
        axes.set_ylabel("Count")
        axes.grid(alpha=0.2)

    def plot_scatter(
        self,
        axes: matplotlib.axes.Axes,
        dataframe: pd.DataFrame,
        x_column: str,
        y_column: str,
        title: str | None = None,
    ) -> None:
        """Plot a scatter chart between two numeric columns."""
        axes.clear()
        x_values = pd.to_numeric(dataframe[x_column], errors="coerce")
        y_values = pd.to_numeric(dataframe[y_column], errors="coerce")
        plot_frame = pd.DataFrame({x_column: x_values, y_column: y_values}).dropna()
        axes.scatter(plot_frame[x_column], plot_frame[y_column], s=50, alpha=0.75, color="#d95f0e")
        axes.set_xlabel(x_column)
        axes.set_ylabel(y_column)
        axes.set_title(title or f"{x_column} vs {y_column}")
        axes.grid(alpha=0.2)

    def plot_missing_values(self, axes: matplotlib.axes.Axes, dataframe: pd.DataFrame) -> None:
        """Plot missing value counts by column."""
        axes.clear()
        missing_counts = dataframe.isna().sum().sort_values(ascending=False)
        missing_counts = missing_counts[missing_counts > 0]
        if missing_counts.empty:
            axes.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=12)
            axes.set_axis_off()
            return
        axes.bar(missing_counts.index, missing_counts.values, color="#6baed6")
        axes.set_title("Missing value count")
        axes.set_ylabel("Missing rows")
        axes.tick_params(axis="x", rotation=35)
        axes.grid(axis="y", alpha=0.2)

    def plot_prediction_scatter(self, axes: matplotlib.axes.Axes, y_true: list[float], y_pred: list[float]) -> None:
        """Plot actual-vs-predicted regression results."""
        axes.clear()
        axes.scatter(y_true, y_pred, s=55, alpha=0.8, color="#31a354")
        if y_true and y_pred:
            lower = min(min(y_true), min(y_pred))
            upper = max(max(y_true), max(y_pred))
            axes.plot([lower, upper], [lower, upper], linestyle="--", color="#636363")
        axes.set_title("Actual vs Predicted")
        axes.set_xlabel("Actual")
        axes.set_ylabel("Predicted")
        axes.grid(alpha=0.2)
