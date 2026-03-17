
# 开发日志


## 设计方案

- 主要内容：Windows Desktop App
- 采用架构：PySide6 / Qt Widgets or QML

### 技术路线

1.MainWindow  

- ProjectPanel        项目/数据集管理
- MoleculeEditor      QWebEngineView 嵌入 Ketcher
- PropertyPanel       分子属性/描述符/预测结果
- DatasetPanel        数据表/筛选/导入导出
- ModelPanel          训练、评估、批量预测
- LogPanel            任务状态/报错/日志

2.Qt ↔ JS Bridge

- QWebChannel
- EditorBridge(QObject)
- Signal: structureChanged(smiles, molblock, metadata)
- Slot: loadStructure(...), saveStructure(...), etc.

3.Python Application Service Layer

- molecule_service.py   分子标准化/校验/格式转换
- descriptor_service.py 描述符/指纹计算
- search_service.py     子结构/相似性检索
- dataset_service.py    导入、清洗、映射、版本管理
- model_service.py      训练、验证、预测、模型保存
- export_service.py     CSV/XLSX/SDF/JSON 导出

4.Data Layer

- SQLite (V1 本地单文件)
- SQLAlchemy / peewee
- files/  模型文件、缓存、原始附件

## 模块搭建

1.UI层

左侧：项目、数据集、模型列表
中间：Ketcher 编辑器
右侧：当前分子详情、RDKit 描述符、预测结果
下方：任务日志、导入记录、报错信息

2.分子编辑层

MoleculeEditorWidget 内部放一个 QWebEngineView，加载本地 index.html，其中嵌入 Ketcher。QWebEngineView 官方就是用来显示和编辑 Web 文档内容的。

3.Qt-JS 通信层

这一层是关键。不要靠 runJavaScript() 到处拼字符串，主通路用 QWebChannel。Qt 官方文档明确说明，WebChannel 可以把 Qt/Python 侧对象暴露给 HTML/JavaScript 客户端，属性、signals、slots 都可以直接访问；在 Qt WebEngine 内嵌页面中，可以直接加载

JS 端收到桥对象后：
用户在 Ketcher 中编辑结构
JS 提取 smiles/molfile
调用 bridge.onStructureEdited(...)
Python 收到后做标准化、校验、入库或触发预测

4.化学服务层

这层单独做成纯 Python 模块，不和 UI 混写。以后你要做单元测试、批处理、CLI 或远程服务都方便。

- molecule_service.py：SMILES/Molfile/InChI 转换，规范化，去盐，合法性校验
- descriptor_service.py：分子描述符、指纹、可视化特征
- search_service.py：精确查重、InChIKey 检索、相似性、子结构
- dataset_service.py：CSV/SDF/XLSX 导入，字段映射，缺失处理
- model_service.py：训练、交叉验证、预测、模型持久化
- report_service.py：结果导出、实验报告、模型报告

5.数据层
目前使用PostgreSQL

## 目录
# chemstudio 项目结构

```text
Lub_Gen/
├─ app.py
├─ requirements.txt
├─ pyproject.toml
├─ resources/
│  ├─ icons/
│  ├─ styles/
│  └─ qt/
│
├─ ui/
│  ├─ main_window.py
│  ├─ project_panel.py
│  ├─ molecule_editor_widget.py
│  ├─ property_panel.py
│  ├─ dataset_panel.py
│  ├─ model_panel.py
│  └─ log_panel.py
│
├─ web/
│  ├─ ketcher/
│  │  ├─ index.html
│  │  ├─ assets/...
│  │  └─ qwebchannel.js
│  └─ bridge.js
│
├─ bridges/
│  ├─ editor_bridge.py
│  └─ app_bridge.py
│
├─ services/
│  ├─ molecule_service.py
│  ├─ descriptor_service.py
│  ├─ search_service.py
│  ├─ dataset_service.py
│  ├─ model_service.py
│  ├─ prediction_service.py
│  ├─ export_service.py
│  └─ report_service.py
│
├─ data/
│  ├─ db.py
│  ├─ models.py
│  ├─ repositories/
│  │  ├─ compound_repo.py
│  │  ├─ dataset_repo.py
│  │  └─ model_repo.py
│  └─ migrations/
│
├─ ml/
│  ├─ featurizers/
│  │  ├─ rdkit_descriptors.py
│  │  └─ fingerprints.py
│  ├─ trainers/
│  │  ├─ regression_trainer.py
│  │  ├─ classification_trainer.py
│  │  └─ validation.py
│  └─ inference/
│     └─ predictor.py
│
├─ workers/
│  ├─ import_worker.py
│  ├─ descriptor_worker.py
│  ├─ train_worker.py
│  └─ predict_worker.py
│
├─ utils/
│  ├─ config.py
│  ├─ logger.py
│  ├─ paths.py
│  └─ exceptions.py
│
├─ tests/
│  ├─ test_molecule_service.py
│  ├─ test_dataset_service.py
│  └─ test_model_service.py
│
└─ packaging/
   ├─ pysidedeploy.spec
   ├─ pyinstaller.spec
   └─ installer/
```

## 关键类关系

```text
   MainWindow
 ├─ ProjectPanel
 ├─ MoleculeEditorWidget
 │   ├─ QWebEngineView
 │   ├─ QWebChannel
 │   └─ EditorBridge
 ├─ PropertyPanel
 ├─ DatasetPanel
 ├─ ModelPanel
 └─ LogPanel

EditorBridge
 └─ calls MoleculeService / DescriptorService / PredictionService

DatasetPanel
 └─ calls DatasetService / ImportWorker

ModelPanel
 └─ calls ModelService / TrainWorker

Data Repositories
 └─ SQLite
```
