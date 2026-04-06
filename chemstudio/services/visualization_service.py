from __future__ import annotations

import html
import json
import os

from chemstudio.utils.config import AppConfig, ensure_runtime_directories
from chemstudio.utils.file_utils import ensure_directory

ensure_runtime_directories()
os.environ.setdefault("MPLCONFIGDIR", str(ensure_directory(AppConfig.LOG_DIR / "matplotlib")))

import matplotlib.axes
import pandas as pd

try:  # pragma: no cover - optional dependency in some environments
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:  # pragma: no cover
    Chem = None
    AllChem = None


class VisualizationService:
    """Centralizes charting and molecule-view rendering helpers for the desktop UI."""

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

    @property
    def rdkit_available(self) -> bool:
        """Whether RDKit-based molecule rendering features are available."""
        return Chem is not None and AllChem is not None

    def generate_3d_molblock(self, smiles: str) -> tuple[str | None, str | None]:
        """Generate a 3D MolBlock from SMILES for viewer rendering."""
        normalized_smiles = smiles.strip()
        if not normalized_smiles:
            return None, "当前记录没有可用的 SMILES。"
        if not self.rdkit_available:
            return None, "当前环境未安装 RDKit，无法生成分子 3D 结构。"

        molecule = Chem.MolFromSmiles(normalized_smiles)
        if molecule is None:
            return None, "无法解析当前 SMILES，不能生成 3D 结构。"

        molecule = Chem.AddHs(molecule)
        embed_parameters = AllChem.ETKDGv3()
        embed_parameters.randomSeed = 0xC0FFEE
        embed_status = AllChem.EmbedMolecule(molecule, embed_parameters)
        if embed_status != 0:
            embed_parameters.useRandomCoords = True
            embed_status = AllChem.EmbedMolecule(molecule, embed_parameters)
        if embed_status != 0:
            return None, "无法为当前分子生成稳定的 3D 构象。"

        try:
            if AllChem.MMFFHasAllMoleculeParams(molecule):
                AllChem.MMFFOptimizeMolecule(molecule, maxIters=500)
            else:
                AllChem.UFFOptimizeMolecule(molecule, maxIters=500)
        except Exception:  # pragma: no cover - best effort fallback for uncommon chemistries
            try:
                AllChem.UFFOptimizeMolecule(molecule, maxIters=500)
            except Exception:
                pass

        return Chem.MolToMolBlock(molecule), None

    def build_molecule_viewer_html(
        self,
        *,
        molblock: str | None = None,
        molecule_name: str = "",
        smiles: str = "",
        error_message: str | None = None,
    ) -> str:
        """Build a self-contained HTML document for the molecule 3D viewer."""
        safe_name = html.escape(molecule_name or "未命名分子")
        safe_smiles = html.escape(smiles or "-")
        viewer_payload = json.dumps(molblock or "")
        message_payload = json.dumps(error_message or "")

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    :root {{
      color-scheme: light;
      --panel-bg: #f8fafc;
      --panel-border: #dbe4ee;
      --text-main: #10233f;
      --text-muted: #5b6b82;
      --accent: #2f6fed;
      --error-bg: #fff4f2;
      --error-border: #f6c6bc;
      --error-text: #9f2d1f;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(47, 111, 237, 0.08), transparent 36%),
        linear-gradient(180deg, #fbfdff 0%, #f3f7fb 100%);
      color: var(--text-main);
    }}

    .shell {{
      display: flex;
      flex-direction: column;
      min-height: 100vh;
      padding: 14px;
      gap: 12px;
    }}

    .meta {{
      padding: 14px 16px;
      border: 1px solid var(--panel-border);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.92);
      box-shadow: 0 10px 28px rgba(15, 35, 63, 0.08);
    }}

    .title {{
      font-size: 16px;
      font-weight: 600;
      line-height: 1.35;
    }}

    .subtitle {{
      margin-top: 6px;
      font-size: 12px;
      line-height: 1.5;
      color: var(--text-muted);
      word-break: break-all;
    }}

    #status {{
      display: none;
      padding: 16px;
      border-radius: 14px;
      border: 1px solid var(--error-border);
      background: var(--error-bg);
      color: var(--error-text);
      line-height: 1.6;
      font-size: 14px;
    }}

    #viewer {{
      flex: 1;
      min-height: 380px;
      border: 1px solid var(--panel-border);
      border-radius: 18px;
      overflow: hidden;
      background:
        radial-gradient(circle at top, rgba(47, 111, 237, 0.08), transparent 42%),
        linear-gradient(180deg, #ffffff 0%, var(--panel-bg) 100%);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
      position: relative;
    }}

    #fallback-canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="meta">
      <div class="title">{safe_name}</div>
      <div class="subtitle">SMILES: {safe_smiles}</div>
    </div>
    <div id="status"></div>
    <div id="viewer"></div>
  </div>
  <script>
    const molBlock = {viewer_payload};
    const errorMessage = {message_payload};
    let canvasState = null;

    function showMessage(message) {{
      const status = document.getElementById("status");
      const viewer = document.getElementById("viewer");
      status.textContent = message;
      status.style.display = "block";
      viewer.style.display = "none";
    }}

    function clearMessage() {{
      const status = document.getElementById("status");
      const viewer = document.getElementById("viewer");
      status.textContent = "";
      status.style.display = "none";
      viewer.style.display = "block";
    }}

    function parseMolBlock(block) {{
      const lines = String(block || "").replace(/\\r/g, "").split("\\n");
      if (lines.length < 4) {{
        return null;
      }}

      const counts = lines[3].trim().split(/\\s+/);
      const atomCount = Number.parseInt(counts[0], 10);
      const bondCount = Number.parseInt(counts[1], 10);
      if (!Number.isFinite(atomCount) || !Number.isFinite(bondCount) || atomCount <= 0) {{
        return null;
      }}

      const atoms = [];
      const bonds = [];

      for (let index = 4; index < 4 + atomCount && index < lines.length; index += 1) {{
        const fields = lines[index].trim().split(/\\s+/);
        if (fields.length < 4) {{
          continue;
        }}
        atoms.push({{
          x: Number.parseFloat(fields[0]),
          y: Number.parseFloat(fields[1]),
          z: Number.parseFloat(fields[2]),
          element: fields[3] || "C",
        }});
      }}

      for (let index = 4 + atomCount; index < 4 + atomCount + bondCount && index < lines.length; index += 1) {{
        const fields = lines[index].trim().split(/\\s+/);
        if (fields.length < 3) {{
          continue;
        }}
        bonds.push({{
          start: Number.parseInt(fields[0], 10) - 1,
          end: Number.parseInt(fields[1], 10) - 1,
          order: Math.max(1, Number.parseInt(fields[2], 10) || 1),
        }});
      }}

      if (!atoms.length) {{
        return null;
      }}

      const center = atoms.reduce(
        (accumulator, atom) => {{
          accumulator.x += atom.x;
          accumulator.y += atom.y;
          accumulator.z += atom.z;
          return accumulator;
        }},
        {{ x: 0, y: 0, z: 0 }}
      );

      center.x /= atoms.length;
      center.y /= atoms.length;
      center.z /= atoms.length;

      let maxDistance = 0;
      for (const atom of atoms) {{
        atom.x -= center.x;
        atom.y -= center.y;
        atom.z -= center.z;
        maxDistance = Math.max(maxDistance, Math.hypot(atom.x, atom.y, atom.z));
      }}

      const normalizer = maxDistance > 0 ? maxDistance : 1;
      for (const atom of atoms) {{
        atom.x /= normalizer;
        atom.y /= normalizer;
        atom.z /= normalizer;
      }}

      return {{ atoms, bonds }};
    }}

    function rotatePoint(point, rotationX, rotationY) {{
      const cosY = Math.cos(rotationY);
      const sinY = Math.sin(rotationY);
      const cosX = Math.cos(rotationX);
      const sinX = Math.sin(rotationX);

      const x1 = point.x * cosY + point.z * sinY;
      const z1 = -point.x * sinY + point.z * cosY;
      const y2 = point.y * cosX - z1 * sinX;
      const z2 = point.y * sinX + z1 * cosX;

      return {{ x: x1, y: y2, z: z2, element: point.element }};
    }}

    function projectPoint(point, width, height, zoom) {{
      const cameraDistance = 4.6;
      const perspective = (Math.min(width, height) * 0.34 * zoom) / Math.max(cameraDistance - point.z, 0.5);
      return {{
        x: width / 2 + point.x * perspective,
        y: height / 2 - point.y * perspective,
        z: point.z,
        scale: perspective,
      }};
    }}

    function getElementColor(element) {{
      const palette = {{
        H: "#d9e2f2",
        C: "#374151",
        N: "#2563eb",
        O: "#dc2626",
        F: "#16a34a",
        P: "#f59e0b",
        S: "#facc15",
        Cl: "#16a34a",
        BR: "#8b4513",
        Br: "#8b4513",
        I: "#7c3aed",
      }};
      return palette[element] || "#64748b";
    }}

    function drawBond(context, firstPoint, secondPoint, bondOrder, firstColor, secondColor, lineWidth) {{
      const dx = secondPoint.x - firstPoint.x;
      const dy = secondPoint.y - firstPoint.y;
      const length = Math.hypot(dx, dy) || 1;
      const offsetX = (-dy / length) * Math.min(4, lineWidth * 0.75);
      const offsetY = (dx / length) * Math.min(4, lineWidth * 0.75);
      const offsets = bondOrder <= 1 ? [0] : bondOrder === 2 ? [-0.8, 0.8] : [-1.6, 0, 1.6];

      for (const offset of offsets) {{
        const ax = firstPoint.x + offsetX * offset;
        const ay = firstPoint.y + offsetY * offset;
        const bx = secondPoint.x + offsetX * offset;
        const by = secondPoint.y + offsetY * offset;
        const midX = (ax + bx) / 2;
        const midY = (ay + by) / 2;

        context.lineCap = "round";
        context.lineWidth = lineWidth;

        context.strokeStyle = firstColor;
        context.beginPath();
        context.moveTo(ax, ay);
        context.lineTo(midX, midY);
        context.stroke();

        context.strokeStyle = secondColor;
        context.beginPath();
        context.moveTo(midX, midY);
        context.lineTo(bx, by);
        context.stroke();
      }}
    }}

    function resetBuiltinView() {{
      if (!canvasState) {{
        return;
      }}
      canvasState.rotationX = -0.6;
      canvasState.rotationY = 0.75;
      canvasState.zoom = 1;
      drawBuiltinViewer();
    }}

    function resizeBuiltinCanvas() {{
      if (!canvasState) {{
        return;
      }}
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(canvasState.container.clientWidth, 320);
      const height = Math.max(canvasState.container.clientHeight, 320);
      canvasState.canvas.width = Math.floor(width * ratio);
      canvasState.canvas.height = Math.floor(height * ratio);
      canvasState.canvas.style.width = `${{width}}px`;
      canvasState.canvas.style.height = `${{height}}px`;
      canvasState.context.setTransform(ratio, 0, 0, ratio, 0, 0);
      drawBuiltinViewer();
    }}

    function drawBuiltinViewer() {{
      if (!canvasState) {{
        return;
      }}

      const width = Math.max(canvasState.container.clientWidth, 320);
      const height = Math.max(canvasState.container.clientHeight, 320);
      const context = canvasState.context;
      context.clearRect(0, 0, width, height);

      const projectedAtoms = canvasState.structure.atoms.map((atom) => {{
        const rotated = rotatePoint(atom, canvasState.rotationX, canvasState.rotationY);
        const projected = projectPoint(rotated, width, height, canvasState.zoom);
        return {{
          ...projected,
          element: atom.element,
        }};
      }});

      const drawItems = [];
      for (const bond of canvasState.structure.bonds) {{
        const firstPoint = projectedAtoms[bond.start];
        const secondPoint = projectedAtoms[bond.end];
        if (!firstPoint || !secondPoint) {{
          continue;
        }}
        drawItems.push({{
          kind: "bond",
          depth: (firstPoint.z + secondPoint.z) / 2,
          firstPoint,
          secondPoint,
          bondOrder: bond.order,
        }});
      }}

      for (const atom of projectedAtoms) {{
        drawItems.push({{
          kind: "atom",
          depth: atom.z,
          atom,
        }});
      }}

      drawItems.sort((left, right) => left.depth - right.depth);

      for (const item of drawItems) {{
        if (item.kind === "bond") {{
          const firstColor = getElementColor(item.firstPoint.element);
          const secondColor = getElementColor(item.secondPoint.element);
          const averageScale = (item.firstPoint.scale + item.secondPoint.scale) / 2;
          const lineWidth = Math.max(2.2, Math.min(7, averageScale * 0.09));
          drawBond(
            context,
            item.firstPoint,
            item.secondPoint,
            item.bondOrder,
            firstColor,
            secondColor,
            lineWidth
          );
        }} else {{
          const radius = Math.max(3.5, Math.min(16, item.atom.scale * 0.12));
          context.fillStyle = getElementColor(item.atom.element);
          context.beginPath();
          context.arc(item.atom.x, item.atom.y, radius, 0, Math.PI * 2);
          context.fill();
          context.lineWidth = 1;
          context.strokeStyle = "rgba(255, 255, 255, 0.82)";
          context.stroke();
        }}
      }}
    }}

    function attachBuiltinInteractions() {{
      if (!canvasState || canvasState.bound) {{
        return;
      }}

      const canvas = canvasState.canvas;
      canvasState.bound = true;

      canvas.addEventListener("pointerdown", (event) => {{
        canvasState.dragging = true;
        canvasState.pointerId = event.pointerId;
        canvasState.lastX = event.clientX;
        canvasState.lastY = event.clientY;
        canvas.setPointerCapture(event.pointerId);
      }});

      canvas.addEventListener("pointermove", (event) => {{
        if (!canvasState.dragging) {{
          return;
        }}
        const deltaX = event.clientX - canvasState.lastX;
        const deltaY = event.clientY - canvasState.lastY;
        canvasState.lastX = event.clientX;
        canvasState.lastY = event.clientY;
        canvasState.rotationY += deltaX * 0.012;
        canvasState.rotationX += deltaY * 0.012;
        drawBuiltinViewer();
      }});

      const stopDrag = (event) => {{
        if (!canvasState.dragging) {{
          return;
        }}
        canvasState.dragging = false;
        if (typeof canvas.releasePointerCapture === "function" && canvasState.pointerId != null) {{
          try {{
            canvas.releasePointerCapture(canvasState.pointerId);
          }} catch (_error) {{
          }}
        }}
        canvasState.pointerId = null;
      }};

      canvas.addEventListener("pointerup", stopDrag);
      canvas.addEventListener("pointerleave", stopDrag);
      canvas.addEventListener("pointercancel", stopDrag);
      canvas.addEventListener("wheel", (event) => {{
        event.preventDefault();
        canvasState.zoom *= event.deltaY < 0 ? 1.08 : 0.92;
        canvasState.zoom = Math.min(2.8, Math.max(0.45, canvasState.zoom));
        drawBuiltinViewer();
      }}, {{ passive: false }});

      canvas.addEventListener("dblclick", () => {{
        resetBuiltinView();
      }});

      if (typeof ResizeObserver !== "undefined") {{
        canvasState.resizeObserver = new ResizeObserver(() => resizeBuiltinCanvas());
        canvasState.resizeObserver.observe(canvasState.container);
      }} else {{
        window.addEventListener("resize", resizeBuiltinCanvas);
      }}
    }}

    function renderBuiltinViewer() {{
      const container = document.getElementById("viewer");
      const structure = parseMolBlock(molBlock);
      if (!structure) {{
        showMessage("MolBlock 解析失败，无法显示当前分子。");
        return false;
      }}

      container.innerHTML = '<canvas id="fallback-canvas"></canvas>';
      const canvas = document.getElementById("fallback-canvas");
      canvasState = {{
        canvas,
        container,
        context: canvas.getContext("2d"),
        structure,
        rotationX: -0.6,
        rotationY: 0.75,
        zoom: 1,
        dragging: false,
        pointerId: null,
        lastX: 0,
        lastY: 0,
        bound: false,
      }};

      if (!canvasState.context) {{
        showMessage("当前环境无法创建 2D 画布，上下文初始化失败。");
        return false;
      }}

      clearMessage();
      attachBuiltinInteractions();
      resizeBuiltinCanvas();
      return true;
    }}

    function renderViewer() {{
      if (errorMessage) {{
        showMessage(errorMessage);
        return;
      }}

      if (!molBlock) {{
        showMessage("当前分子没有可显示的 3D 结构。");
        return;
      }}

      clearMessage();
      renderBuiltinViewer();
    }}

    window.resetView = function() {{
      resetBuiltinView();
    }};

    document.addEventListener("DOMContentLoaded", renderViewer);
  </script>
</body>
</html>
"""
