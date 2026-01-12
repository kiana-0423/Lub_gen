#include "mainwindow.h"

#include "csvparser.h"

#include <QFileDialog>
#include <QGridLayout>
#include <QHeaderView>
#include <QMessageBox>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlError>

MainWindow::MainWindow(QWidget* parent)
    : QWidget(parent),
      csvPathEdit_(new QLineEdit(this)),
      dbPathEdit_(new QLineEdit(this)),
      openCsvButton_(new QPushButton("Open CSV", this)),
      browseDbButton_(new QPushButton("Browse...", this)),
      importButton_(new QPushButton("Import to SQLite", this)),
      headerCheck_(new QCheckBox("First row is header", this)),
      previewTable_(new QTableWidget(this)),
      statusLabel_(new QLabel(this)) {
    setWindowTitle("CSV Collector");
    resize(900, 600);

    csvPathEdit_->setReadOnly(true);
    dbPathEdit_->setPlaceholderText("SQLite database path");
    headerCheck_->setChecked(true);

    previewTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    previewTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    previewTable_->horizontalHeader()->setStretchLastSection(true);

    auto* layout = new QGridLayout(this);
    layout->addWidget(new QLabel("CSV File:", this), 0, 0);
    layout->addWidget(csvPathEdit_, 0, 1);
    layout->addWidget(openCsvButton_, 0, 2);
    layout->addWidget(new QLabel("SQLite DB:", this), 1, 0);
    layout->addWidget(dbPathEdit_, 1, 1);
    layout->addWidget(browseDbButton_, 1, 2);
    layout->addWidget(headerCheck_, 2, 0, 1, 3);
    layout->addWidget(previewTable_, 3, 0, 1, 3);
    layout->addWidget(importButton_, 4, 2);
    layout->addWidget(statusLabel_, 4, 0, 1, 2);
    setLayout(layout);

    connect(openCsvButton_, &QPushButton::clicked, this, &MainWindow::onOpenCsv);
    connect(browseDbButton_, &QPushButton::clicked, this, &MainWindow::onBrowseDb);
    connect(importButton_, &QPushButton::clicked, this, &MainWindow::onImport);
    connect(headerCheck_, &QCheckBox::toggled, this, &MainWindow::onHeaderToggled);

    statusLabel_->setText("Load a CSV file to start.");
}

void MainWindow::onOpenCsv() {
    const QString path = QFileDialog::getOpenFileName(
        this, "Open CSV", QString(), "CSV Files (*.csv);;All Files (*)");
    if (path.isEmpty()) {
        return;
    }
    loadCsv(path);
}

void MainWindow::onBrowseDb() {
    const QString path = QFileDialog::getSaveFileName(
        this, "Select SQLite Database", QString(), "SQLite DB (*.sqlite *.db);;All Files (*)");
    if (!path.isEmpty()) {
        dbPathEdit_->setText(path);
    }
}

void MainWindow::onHeaderToggled() {
    updatePreview();
}

void MainWindow::loadCsv(const QString& path) {
    QString error;
    QVector<QStringList> rows = CsvParser::parseFile(path, &error);
    if (!error.isEmpty()) {
        QMessageBox::warning(this, "CSV Error", error);
        return;
    }
    rows_ = std::move(rows);
    csvPathEdit_->setText(path);

    if (dbPathEdit_->text().trimmed().isEmpty()) {
        QString dbPath = path;
        if (dbPath.endsWith(".csv", Qt::CaseInsensitive)) {
            dbPath.chop(4);
        }
        dbPath += ".sqlite";
        dbPathEdit_->setText(dbPath);
    }

    updatePreview();
}

void MainWindow::updatePreview() {
    previewTable_->clear();

    if (rows_.isEmpty()) {
        previewTable_->setRowCount(0);
        previewTable_->setColumnCount(0);
        statusLabel_->setText("CSV is empty.");
        return;
    }

    QStringList headers;
    int startRow = 0;
    int columnCount = 0;

    if (headerCheck_->isChecked()) {
        headers = rows_.first();
        startRow = 1;
        columnCount = headers.size();
    } else {
        columnCount = rows_.first().size();
        headers = makeHeaders(columnCount);
    }

    for (const auto& row : rows_) {
        columnCount = qMax(columnCount, row.size());
    }
    if (headers.size() < columnCount) {
        headers = makeHeaders(columnCount);
    }

    previewTable_->setColumnCount(columnCount);
    previewTable_->setHorizontalHeaderLabels(headers);

    const int previewRows = qMin(100, rows_.size() - startRow);
    previewTable_->setRowCount(previewRows);

    for (int r = 0; r < previewRows; ++r) {
        const QStringList row = rows_.at(startRow + r);
        for (int c = 0; c < columnCount; ++c) {
            const QString value = c < row.size() ? row.at(c) : QString();
            previewTable_->setItem(r, c, new QTableWidgetItem(value));
        }
    }

    statusLabel_->setText(QString("Loaded %1 rows, %2 columns.")
                              .arg(rows_.size() - startRow)
                              .arg(columnCount));
}

QStringList MainWindow::makeHeaders(int columnCount) const {
    QStringList headers;
    headers.reserve(columnCount);
    for (int i = 0; i < columnCount; ++i) {
        headers.push_back(QString("col_%1").arg(i + 1));
    }
    return headers;
}

QStringList MainWindow::sanitizeHeaders(const QStringList& input) const {
    QStringList output;
    output.reserve(input.size());
    QSet<QString> seen;

    for (int i = 0; i < input.size(); ++i) {
        QString name = input.at(i).trimmed();
        QString cleaned;
        for (const QChar c : name) {
            if (c.isLetterOrNumber() || c == '_') {
                cleaned.append(c);
            } else {
                cleaned.append('_');
            }
        }
        if (cleaned.isEmpty()) {
            cleaned = QString("col_%1").arg(i + 1);
        }
        if (!cleaned.isEmpty() && cleaned.at(0).isDigit()) {
            cleaned.prepend("c_");
        }
        QString unique = cleaned;
        int suffix = 1;
        while (seen.contains(unique)) {
            unique = QString("%1_%2").arg(cleaned).arg(++suffix);
        }
        seen.insert(unique);
        output.push_back(unique);
    }
    return output;
}

bool MainWindow::ensureDbPath() {
    QString path = dbPathEdit_->text().trimmed();
    if (!path.isEmpty()) {
        return true;
    }
    if (csvPathEdit_->text().trimmed().isEmpty()) {
        QMessageBox::information(this, "Missing Path", "Please choose a CSV file first.");
        return false;
    }
    QString dbPath = csvPathEdit_->text();
    if (dbPath.endsWith(".csv", Qt::CaseInsensitive)) {
        dbPath.chop(4);
    }
    dbPath += ".sqlite";
    dbPathEdit_->setText(dbPath);
    return true;
}

void MainWindow::onImport() {
    if (rows_.isEmpty()) {
        QMessageBox::information(this, "No Data", "Load a CSV file first.");
        return;
    }
    if (!ensureDbPath()) {
        return;
    }

    QStringList headers;
    int startRow = 0;
    if (headerCheck_->isChecked()) {
        headers = rows_.first();
        startRow = 1;
    } else {
        headers = makeHeaders(rows_.first().size());
    }

    int columnCount = headers.size();
    for (const auto& row : rows_) {
        columnCount = qMax(columnCount, row.size());
    }
    if (headers.size() < columnCount) {
        headers = makeHeaders(columnCount);
    }
    headers = sanitizeHeaders(headers);

    if (QSqlDatabase::contains("import")) {
        QSqlDatabase::removeDatabase("import");
    }
    QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE", "import");
    db.setDatabaseName(dbPathEdit_->text().trimmed());
    if (!db.open()) {
        QMessageBox::critical(this, "DB Error", db.lastError().text());
        return;
    }

    QSqlQuery query(db);
    if (!query.exec("DROP TABLE IF EXISTS records")) {
        QMessageBox::critical(this, "DB Error", query.lastError().text());
        return;
    }

    QStringList columnDefs;
    columnDefs.reserve(headers.size());
    for (const auto& name : headers) {
        columnDefs.push_back(QString("`%1` TEXT").arg(name));
    }
    const QString createSql = QString("CREATE TABLE records (%1)").arg(columnDefs.join(", "));
    if (!query.exec(createSql)) {
        QMessageBox::critical(this, "DB Error", query.lastError().text());
        return;
    }

    QStringList placeholders;
    placeholders.reserve(headers.size());
    for (int i = 0; i < headers.size(); ++i) {
        placeholders.push_back("?");
    }
    const QString insertSql = QString("INSERT INTO records (%1) VALUES (%2)")
                                  .arg(headers.join(", "))
                                  .arg(placeholders.join(", "));
    if (!query.prepare(insertSql)) {
        QMessageBox::critical(this, "DB Error", query.lastError().text());
        return;
    }

    db.transaction();
    int inserted = 0;
    for (int i = startRow; i < rows_.size(); ++i) {
        const QStringList row = rows_.at(i);
        query.clear();
        for (int c = 0; c < headers.size(); ++c) {
            query.addBindValue(c < row.size() ? row.at(c) : QString());
        }
        if (!query.exec()) {
            db.rollback();
            QMessageBox::critical(this, "DB Error", query.lastError().text());
            return;
        }
        ++inserted;
    }
    db.commit();

    statusLabel_->setText(QString("Imported %1 rows into %2.")
                              .arg(inserted)
                              .arg(dbPathEdit_->text().trimmed()));
    QMessageBox::information(this, "Import Complete", statusLabel_->text());
}
