# ChemStudio

ChemStudio 是一个面向材料/化学数据管理、分子设计、配方设计与性能预测的桌面软件 MVP。项目采用模块化分层结构，使用 `PySide6 + SQLite + pandas + scikit-learn + matplotlib` 实现可运行的多页面桌面应用。

## 功能概览

- 首页 `HomePage`
  - 提供数据导入与可视化、分子设计、配方设计三个功能入口
- 数据导入与可视化 `DataPage`
  - 导入 CSV / Excel
  - 数据写入 SQLite
  - 表格浏览、检索、删除、刷新
  - 分布图、散点图、缺失值统计
- 分子设计 `MoleculeDesignPage`
  - 从数据库读取训练数据
  - 选择目标性能列和回归模型
  - 输出 `R² / MAE / RMSE`
  - 真实值 vs 预测值散点图
  - 模型保存 / 加载
  - 单分子输入预测，支持 RDKit 描述符兼容接口
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
│  │  ├─ trainer.py
│  │  ├─ predictor.py
│  │  └─ metrics.py
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

如果需要 RDKit 描述符自动生成：

```bash
pip install rdkit
```

如果需要 XGBoost：

```bash
pip install xgboost
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

## 说明

- SQLite 数据库默认位于 `chemstudio/resources/chemstudio.sqlite`，可通过 `CHEMSTUDIO_DATABASE_PATH` 覆盖
- 已训练模型默认保存到 `chemstudio/resources/saved_models/`
- RDKit 和 XGBoost 都是兼容式可选依赖，不安装也可以运行主体功能
- 当前配方特征工程采用简单加权平均策略，代码结构已为后续更复杂规则预留扩展点
