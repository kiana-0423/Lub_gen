#pragma once

#include <QString>
#include <QStringList>
#include <QVector>

class CsvParser {
public:
    static QVector<QStringList> parseFile(const QString& path, QString* error);

private:
    static QStringList parseLine(const QString& line);
};
