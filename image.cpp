#include "image.h"
#include "imageset.h"

#include <fstream>
#include <sstream>
#include <string>

#include <QDebug>
#include <iostream>

using namespace std;

Image::Image(ImageSet *parent, QString path) : m_parentSet(parent), m_path(path)
{
    loaded = false;
}

int Image::load()
{
    int row = 0, col = 0, numrows = 0, numcols = 0, pixelmax = 0;
    if(m_path.isEmpty() || m_parentSet->getPath().isEmpty())
    {
        //qDebug()<<"Calea catre fisier este goala!";
        return -1; //fail
    }
    QString fullPath = m_parentSet->getPath().append("/").append(m_path);
    ifstream infile(fullPath.toUtf8());
    stringstream ss;
    string inputLine = "";
    getline(infile,inputLine);
    if(inputLine.compare("P5") != 0)
    {
        //qDebug()<<"Fisier in format gresit." << fullPath;
        return -1; //fail
    }

    ss << infile.rdbuf();
    ss >> numcols >> numrows  >> pixelmax;
    if(!numcols && !numrows)
    {
        //qDebug()<<"Coloane/randuri nu pot fi citite.";
        return -1;
    }
    // am inceput incarcarea efectiva
    emit startedLoading();

    unsigned char *array = new unsigned char[numrows*numcols];
    for(int i = 0; i < numrows*numcols; i++)
            ss >> array[i];
    infile.close();
    m_height = numrows;
    m_width = numcols;
    m_array = new float[m_width*m_height];
    for(int i=0;i<m_width*m_height;i++)
    {
        m_array[i] = array[i];
    }

    //am terminat incarcarea
    loaded = true;
    emit finishedLoading();
    //qDebug()<<"Am terminat incarcarea.";
    return 0; //success
}

QImage Image::toQImage()
{
    //qDebug()<<"Redau imaginea.";
    /*
    for(int row = 0; row < m_height; row++)
    {
        for(int col = 0; col<m_width; col++)
        {
            cout<<QString::number((int)m_array[row*col]).toStdString()<<' ';
        }
        cout<<"\n";
    }*/
    return QImage(m_parentSet->getPath().append("/").append(m_path));
}

Image::~Image()
{
    delete(m_array);
    delete(m_parentSet);
}

int Image::width()
{
    return m_width;
}

int Image::height()
{
    return m_height;
}

float *Image::getArray()
{
    return m_array;
}
