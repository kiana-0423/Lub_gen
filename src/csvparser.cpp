#include "csvparser.h"

#include <QFile>
#include <QTextStream>

QVector<QStringList> CsvParser::parseFile(const QString& path, QString* error) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (error) {
            *error = QString("Failed to open file: %1").arg(file.errorString());
        }
        return {};
    }

    QVector<QStringList> rows;
    QTextStream in(&file);
    while (!in.atEnd()) {
        QString line = in.readLine();
        if (!line.isEmpty() && line.endsWith('\r')) {
            line.chop(1);
        }
        rows.push_back(parseLine(line));
    }
    return rows;
}

QStringList CsvParser::parseLine(const QString& line) {
    QStringList fields;
    QString field;
    bool inQuotes = false;

    for (int i = 0; i < line.size(); ++i) {
        const QChar c = line.at(i);
        if (inQuotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line.at(i + 1) == '"') {
                    field.append('"');
                    ++i;
                } else {
                    inQuotes = false;
                }
            } else {
                field.append(c);
            }
        } else {
            if (c == '"') {
                inQuotes = true;
            } else if (c == ',') {
                fields.push_back(field);
                field.clear();
            } else {
                field.append(c);
            }
        }
    }
    fields.push_back(field);
    return fields;
}
