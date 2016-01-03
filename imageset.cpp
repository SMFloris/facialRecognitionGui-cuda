#include "imageset.h"
#include "math.h"
#include "image.h"

#include <stdio.h>
#include <bitset>
#include <QDir>
#include <QDebug>
using namespace std;

ImageSet::ImageSet(QString numeTeste, int idSet, int maxCount) : m_name(numeTeste), m_id(idSet), m_maxCount(maxCount)
{
    m_path = QString(QDir::currentPath()).append("/").append(m_name).append("/s").append(QString::number(m_id));
}

void ImageSet::load()
{
    emit startedLoading(m_id);
        QDir recoredDir(m_path);
        QStringList images = recoredDir.entryList();
        QString lastImage;
        //qDebug()<< m_path;// << recoredDir.absoluteFilePath(images.first());
        foreach (QString image, images) {
            //printf("%d\n", m_id);
            if(image.endsWith(".pgm")){
                //imaginea este ok
                //qDebug() << "Adaug imagine: " <<image;
                addImage(image);
            }
        }
    emit finishedLoading(10);
}

void ImageSet::addImage(QString path)
{
    Image *img = new Image(this,path);
    imagini.append(img);
}

int ImageSet::loadUp(int id)
{
    if(imagini.at(id)->load()==0)
        return 0;
    return -1; //fail
}

void ImageSet::freeUp(int id)
{
    imagini.removeAt(id);
}

QImage ImageSet::getQImage(int id)
{
    return imagini.at(id)->toQImage();
}

Image *ImageSet::getImage(int id)
{
    return imagini.at(id);
}

QString ImageSet::getPath()
{
    return m_path;
}

size_t ImageSet::getSize()
{
    return imagini.count()*sizeof(char)*imagini.first()->width()*imagini.first()->height();
}

int ImageSet::count()
{
    return imagini.count();
}

float *ImageSet::loadedToFloat()
{
    float *rez = (float*) malloc(getSize()*sizeof(float));
    int counter = 0;

    QList<Image*>::iterator i;
    for (i = imagini.begin(); i != imagini.end(); ++i)
        if((*i)->loaded)
            memcpy(rez+(counter++)*10304,(*i)->getArray(),10304*sizeof(float));

    return rez;
}
