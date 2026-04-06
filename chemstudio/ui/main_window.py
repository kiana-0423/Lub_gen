from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QStackedWidget

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.services import (
    DataImportService,
    FeatureService,
    FormulaService,
    ModelService,
    VisualizationService,
)
from chemstudio.ui.data_page import DataPage
from chemstudio.ui.formula_design_page import FormulaDesignPage
from chemstudio.ui.home_page import HomePage
from chemstudio.ui.molecule_design_page import MoleculeDesignPage
from chemstudio.utils.config import AppConfig


class MainWindow(QMainWindow):
    """Top-level window that owns services and central navigation."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        super().__init__()
        self.db_manager = db_manager
        self.data_import_service = DataImportService(db_manager)
        self.visualization_service = VisualizationService()
        self.feature_service = FeatureService(db_manager)
        self.model_service = ModelService(db_manager, self.feature_service)
        self.formula_service = FormulaService(db_manager)

        self.setWindowTitle(AppConfig.APP_NAME)
        self.resize(AppConfig.WINDOW_WIDTH, AppConfig.WINDOW_HEIGHT)

        self.page_stack = QStackedWidget()
        self.setCentralWidget(self.page_stack)

        self.pages = self._build_pages()
        self._connect_navigation()
        self._build_toolbar()
        self.navigate("home")

    def _build_pages(self) -> dict[str, object]:
        home_page = HomePage()
        data_page = DataPage(self.db_manager, self.data_import_service, self.visualization_service)
        molecule_page = MoleculeDesignPage(
            self.db_manager,
            self.feature_service,
            self.model_service,
            self.visualization_service,
        )
        formula_page = FormulaDesignPage(self.db_manager, self.formula_service, self.model_service)

        pages = {
            "home": home_page,
            "data": data_page,
            "molecule": molecule_page,
            "formula": formula_page,
        }
        for page in pages.values():
            self.page_stack.addWidget(page)
        return pages

    def _connect_navigation(self) -> None:
        self.pages["home"].navigate_requested.connect(self.navigate)
        self.pages["data"].home_requested.connect(lambda: self.navigate("home"))
        self.pages["molecule"].home_requested.connect(lambda: self.navigate("home"))
        self.pages["formula"].home_requested.connect(lambda: self.navigate("home"))

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Navigation")
        toolbar.setMovable(False)
        toolbar.addAction("首页", lambda: self.navigate("home"))
        toolbar.addAction("数据导入", lambda: self.navigate("data"))
        toolbar.addAction("分子设计", lambda: self.navigate("molecule"))
        toolbar.addAction("配方设计", lambda: self.navigate("formula"))

    def navigate(self, page_key: str) -> None:
        """Switch to a page and trigger a page-level refresh hook."""
        page = self.pages[page_key]
        self.page_stack.setCurrentWidget(page)
        if hasattr(page, "refresh_page"):
            page.refresh_page()
        self.statusBar().showMessage(f"当前页面: {page_key}", 3000)
