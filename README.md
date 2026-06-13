# ChemStudio v2.0.0

ChemStudio 是一个面向润滑材料/化学数据管理、分子建模、分子设计、配方设计与性能预测的桌面软件。项目采用模块化分层结构，使用 `PySide6 + SQLite + pandas + RDKit/Mordred + scikit-learn + matplotlib` 构建多页面桌面应用。

当前版本：`v2.0.0`（Python 包版本：`2.0.0`）。

## v2.0.0 亮点

- 新增分子编辑器页面：支持小分子绘制、SMILES 导入/导出、MOL/SDF/PDB 导出，并可直接导入数据库。
- 新增 `draw/` 核心包，承载分子绘图、命令栈、编辑画布和 RDKit 化学服务。
- 新增润滑材料专有数据结构：基础油、添加剂、润滑性能指标、配方组件、配方测试结果和添加剂/基础油相容性。
- 数据导入支持材料类型、基础油类别、添加剂类别和常见润滑性能字段的中英文别名识别。
- `molecules.material_type_id` 支持材料类型外键筛选，替代仅依赖参数表的材料类型过滤。
- 数据页面支持按基础油/添加剂筛选，并在选中添加剂时展示基础油相容性信息。
- 配方设计支持基础油/添加剂角色标注，并在添加剂与基础油相容性较差时显示黄色警告。
- 配方组件写入结构化关系表 `formula_components`，不再只保存在 JSON 字段中。
- 历史 `property_data` 中的润滑属性会同步迁移到 `lubricant_properties`。
- 继续保留 Mordred 描述符、特征筛选、SHAP 解释、模型训练/预测和 Repository 数据访问层。

## 功能概览

- 首页 `HomePage`
  - 提供数据导入与可视化、分子设计、分子编辑器、配方设计四个功能入口。
  - 四个模块按两行两列展示，并支持小窗口滚动查看。

- 数据导入与可视化 `DataPage`
  - 导入 CSV / Excel / JSON。
  - 自动识别 SMILES、分子名称、材料类型、基础油/添加剂类别、实验条件和润滑性能字段。
  - 数据写入 SQLite，并支持按材料类型筛选。
  - 表格浏览、检索、删除、刷新。
  - 分子 3D 可视化。
  - 导出当前筛选后的 Mordred 特征 CSV。
  - 选中添加剂时展示其与基础油的相容性评分、溶解度和备注，并支持编辑。

- 分子编辑器 `MoleculeEditorPage`
  - 绘制小分子结构。
  - 支持选择、原子、键、删除等编辑工具。
  - 支持 C、N、O、S、P、Cl、Br、F 等常用元素。
  - 支持单键、双键、三键和芳香键。
  - 支持撤销、重做、清除画布、缩放、生成 2D 坐标和展开显式氢。
  - 支持 SMILES 导入、生成和复制。
  - 支持 MOL / SDF / PDB 导出。
  - 支持将当前绘制分子直接保存到 ChemStudio 数据库。

- 分子设计 `MoleculeDesignPage`
  - 从数据库读取训练数据。
  - 自动推断回归/分类任务。
  - 选择目标性能列、模型、交叉验证和超参数搜索策略。
  - 支持 Mordred 高维特征筛选，生成可追溯报告。
  - 输出 `R² / MAE / RMSE` 或分类指标。
  - 真实值 vs 预测值散点图。
  - 模型保存 / 安全加载。
  - 单分子 SMILES 输入预测，支持 RDKit/Mordred 描述符。

- 配方设计 `FormulaDesignPage`
  - 选择多个分子并输入配比。
  - 标注每个组分角色：基础油或添加剂。
  - 保存结构化配方组件到 `formula_components`。
  - 根据基础油/添加剂角色构建配方特征。
  - 自动检查添加剂与基础油相容性，低评分或不溶时显示警告。
  - 训练配方模型并预测配方性能。
  - 保存配方记录和预测测试结果。

## 项目结构

```text
project_root/
├─ app.py
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ draw/
│  ├─ core/
│  ├─ editor/
│  ├─ commands/
│  ├─ chemistry_services/
│  └─ ui/
├─ chemstudio/
│  ├─ app.py
│  ├─ constants.py
│  ├─ database/
│  │  ├─ db_manager.py
│  │  ├─ models.py
│  │  └─ repositories/
│  ├─ ml/
│  │  ├─ feature_selection.py
│  │  ├─ trainer.py
│  │  ├─ predictor.py
│  │  └─ metrics.py
│  ├─ validation/
│  ├─ resources/
│  │  ├─ mock_materials.csv
│  │  ├─ chemstudio.sqlite
│  │  └─ chemstudio_mvp.sqlite
│  ├─ services/
│  │  ├─ data_import_service.py
│  │  ├─ feature_service.py
│  │  ├─ formula_service.py
│  │  ├─ formula_test_service.py
│  │  ├─ model_service.py
│  │  └─ visualization_service.py
│  ├─ ui/
│  │  ├─ main_window.py
│  │  ├─ home_page.py
│  │  ├─ data_page.py
│  │  ├─ molecule_design_page.py
│  │  ├─ molecule_editor_page.py
│  │  ├─ molecule_editor_inspector.py
│  │  ├─ formula_design_page.py
│  │  └─ widgets/
│  └─ utils/
│     ├─ config.py
│     ├─ file_utils.py
│     └─ logger.py
└─ examples/
   └─ chemstudio_import_sample.csv
```

## 安装

建议使用 Python 3.11+ 虚拟环境。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果需要 SHAP 模型解释与可视化：

```bash
pip install -e ".[explain]"
```

如果需要 XGBoost：

```bash
pip install -e ".[boost]"
```

开发环境建议安装开发依赖：

```bash
pip install -e ".[dev]"
```

## 运行

```bash
python app.py
```

或者：

```bash
python -m chemstudio
```

首次启动会自动初始化 SQLite 数据库。若数据库为空，会导入 `chemstudio/resources/mock_materials.csv` 作为演示数据。

## 示例数据

示例 CSV 位于：

```text
examples/chemstudio_import_sample.csv
```

该文件包含基础油、添加剂、材料类型、类别、添加量、实验条件和润滑性能字段，可直接用于导入测试。

## 测试

```bash
pytest -q
```

当前测试覆盖数据库、Repository、导入服务、模型服务、配方服务、特征选择、验证层、日志配置和关键 UI 页面。

## 说明

- SQLite 数据库默认位于 `chemstudio/resources/chemstudio.sqlite`，可通过 `CHEMSTUDIO_DATABASE_PATH` 覆盖。
- 已训练模型默认保存到 `chemstudio/resources/saved_models/`。
- RDKit 和 Mordred 用于分子结构解析、编辑器化学转换与描述符计算。
- SHAP 是可选依赖，用于训练后的全局特征重要性、摘要图和单样本局部解释。
- XGBoost 是兼容式可选依赖，不安装也可以运行主体功能。
- 日志级别可通过 `CHEMSTUDIO_LOG_LEVEL` 覆盖，例如 `DEBUG`、`INFO`、`WARNING`。
