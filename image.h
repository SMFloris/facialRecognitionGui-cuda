#ifndef IMAGE_H
#define IMAGE_H

#include <QObject>
#include <QImage>

class ImageSet;

class Image : public QObject
{
    Q_OBJECT
public:
    Image(ImageSet *parentSet, QString path);
    ~Image();

    bool loaded;

    int width();
    int height();

    float *getArray();

signals:
    void startedLoading();
    void finishedLoading();

public slots:
    int load();
    QImage toQImage();


private:
    ImageSet *m_parentSet;
    QString m_path;
    float *m_array;
    int m_width, m_height;
};

#endif // IMAGE_H
