#ifndef IMAGESET_H
#define IMAGESET_H

#include <QObject>
#include <QImage>
#include <QList>

class Image;

class ImageSet : public QObject
{
    Q_OBJECT
public:
    ImageSet(QString numeTeste, int idSet, int maxCount);

    QString getPath();
    size_t getSize();
    int count();

    float *loadedToFloat();

signals:
    void startedLoading(int idSet);
    void finishedLoading(int totalBytesLoaded);

public slots:
    void load();
    void addImage(QString path);
    int loadUp(int id);
    void freeUp(int id);
    QImage getQImage(int id);
    Image* getImage(int id);

private:
    QList<Image*> imagini;
    //path catre set
    QString m_path;
    //nume set
    QString m_name;
    //id_set
    unsigned int m_id;
    //numarul maxim de poze
    const int m_maxCount;

};

#endif // IMAGESET_H
