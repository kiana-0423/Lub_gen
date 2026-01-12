#pragma once

#include <QCheckBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QTableWidget>
#include <QVector>
#include <QStringList>
#include <QWidget>

class MainWindow : public QWidget {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);

private slots:
    void onOpenCsv();
    void onBrowseDb();
    void onImport();
    void onHeaderToggled();

private:
    void loadCsv(const QString& path);
    void updatePreview();
    QStringList makeHeaders(int columnCount) const;
    QStringList sanitizeHeaders(const QStringList& input) const;
    bool ensureDbPath();

    QLineEdit* csvPathEdit_;
    QLineEdit* dbPathEdit_;
    QPushButton* openCsvButton_;
    QPushButton* browseDbButton_;
    QPushButton* importButton_;
    QCheckBox* headerCheck_;
    QTableWidget* previewTable_;
    QLabel* statusLabel_;

    QVector<QStringList> rows_;
};
