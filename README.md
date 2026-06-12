# ChemStudio v1.1

ChemStudio 是一个面向材料/化学数据管理、分子设计、配方设计与性能预测的桌面软件 MVP。项目采用模块化分层结构，使用 `PySide6 + SQLite + pandas + RDKit/Mordred + scikit-learn + matplotlib` 实现可运行的多页面桌面应用。

当前版本：`v1.1`（Python 包版本：`1.1.0`）。

## v1.1 亮点

- 新增 Mordred 1800+ 分子描述符计算与 CSV 导出能力
- 新增多阶段特征筛选：方差、缺失值、共线性、互信息、模型驱动筛选
- 新增可选 SHAP 模型解释：全局特征重要性、摘要图和单样本局部贡献
- 分子设计与配方设计训练流程异步化，减少 UI 卡顿
- 数据访问层引入 Repository：`MoleculeRepository`、`DescriptorRepository`、`ModelRepository`、`FormulaRepository`、`PredictionRepository`
- 增加统一验证层，集中校验 SMILES、分子名称、配方比例、目标列和特征列
- 增强 SQLite 连接配置，启用 WAL、busy timeout 和更安全的排序字段白名单
- 增加 pre-commit、CI、pytest-qt 等工程化配置

## 功能概览

- 首页 `HomePage`
  - 提供数据导入与可视化、分子设计、配方设计三个功能入口
- 数据导入与可视化 `DataPage`
  - 导入 CSV / Excel / JSON
  - 数据写入 SQLite
  - 表格浏览、检索、删除、刷新
  - 分布图、散点图、缺失值统计
  - 导出当前筛选后的 Mordred 特征 CSV
- 分子设计 `MoleculeDesignPage`
  - 从数据库读取训练数据
  - 自动推断回归/分类任务
  - 选择目标性能列、模型、交叉验证和超参数搜索策略
  - 支持 Mordred 高维特征筛选，生成可追溯报告
  - 输出 `R² / MAE / RMSE` 或分类指标
  - 真实值 vs 预测值散点图
  - 模型保存 / 安全加载
  - 单分子 SMILES 输入预测，支持 RDKit/Mordred 描述符
- 配方设计 `FormulaDesignPage`
  - 选择多个分子并输入配比
  - 按加权平均策略生成配方特征
  - 加载模型后预测配方性能
  - 保存配方记录到数据库

## 项目结构

```text
project_root/
├─ app.py
├─ README.md
├─ requirements.txt
├─ chemstudio/
│  ├─ app.py
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
│  │  └─ mock_materials.csv
│  ├─ services/
│  │  ├─ data_import_service.py
│  │  ├─ visualization_service.py
│  │  ├─ feature_service.py
│  │  ├─ model_service.py
│  │  └─ formula_service.py
│  ├─ ui/
│  │  ├─ main_window.py
│  │  ├─ home_page.py
│  │  ├─ data_page.py
│  │  ├─ molecule_design_page.py
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

如果需要 XGBoost：

```bash
pip install xgboost
```

如果需要 SHAP 模型解释与可视化：

```bash
pip install -e ".[explain]"
```

开发环境建议安装开发依赖并启用 pre-commit：

```bash
pip install -e ".[dev]"
pre-commit install
```

## 运行

```bash
python app.py
```

或者：

```bash
python -m chemstudio
```

首次启动会自动将 `chemstudio/resources/mock_materials.csv` 导入到 SQLite 数据库，便于直接测试训练、预测和配方流程。

## 测试

```bash
pytest -q
```

当前测试覆盖数据库、Repository、导入服务、模型服务、特征选择、验证层、日志配置和关键 UI 页面。

## 说明

- SQLite 数据库默认位于 `chemstudio/resources/chemstudio.sqlite`，可通过 `CHEMSTUDIO_DATABASE_PATH` 覆盖
- 已训练模型默认保存到 `chemstudio/resources/saved_models/`
- RDKit 和 Mordred 用于分子结构解析与描述符计算
- SHAP 是可选依赖，用于训练后的全局特征重要性、摘要图和单样本局部解释
- XGBoost 是兼容式可选依赖，不安装也可以运行主体功能
- 当前配方特征工程采用简单加权平均策略，代码结构已为后续更复杂规则预留扩展点
- 日志级别可通过 `CHEMSTUDIO_LOG_LEVEL` 覆盖，例如 `DEBUG`、`INFO`、`WARNING`
