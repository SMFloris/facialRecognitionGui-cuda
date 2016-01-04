#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QPixmap>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QDebug>
#include <QImage>

#include "imageset.h"
#include "image.h"

#include "math.h"

#define KVECINI 3

extern void cumSum(float *out, float *in, unsigned int n);
extern void test1(float *a, float *b, unsigned long n);

extern void cityblockNormAsync(float *a, float *b, unsigned long n, float *result, unsigned long nrPoze);
extern void euclideanNormAsync(float *a, float *b, unsigned long n, float *result, unsigned long nrPoze);
extern void cosNormAsync(float *a, float *b, unsigned long n, float *result, unsigned long nrPoze);


float minimum(float *a, int n, int count=0)
{
    float min;
    if(count == 0)
    {
        min = a[0];
        for(int i=0;i<n;i++)
        {
            if(!isinf(a[i]) && a[i]<min)
                min = a[i];
        }
    } else
    {
        for(int i=0;i<n-1;i++)
            for(int j=i+1;j<n;j++)
            {
                float aux;
                if(a[j]<a[i])
                {
                    aux = a[j];
                    a[j] = a[i];
                    a[i] = aux;
                }
            }
        for(int i=0;i<n;i++)
        {
            min += a[i];
        }
        min /= count;
    }
    return min;
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_actionQuit_triggered()
{
    this->close();
}

/*
void MainWindow::on_select_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("Open Image"), "/home", tr("Image Files (*.png *.jpg *.bmp *.pgm)"));
    QGraphicsScene* scene = new QGraphicsScene();
    QGraphicsView* view = new QGraphicsView(scene);
    QImage img(fileName);
    QGraphicsPixmapItem* item = new QGraphicsPixmapItem(QPixmap::fromImage(img));
    scene->addItem(item);
    ui->img2->setScene(scene);
}
*/

void MainWindow::on_start_clicked()
{
    int idSetImgCautata = 4;
    int idImgCautata = 5;
    float *imagineCautata = (float*) malloc(10304*sizeof(float));
    if(ui->nn->isChecked())
    {
        if(!(ui->norm1->isChecked() || ui->norm2->isChecked() || ui->norm3->isChecked() || ui->norm4->isChecked()))
            return;

        float min;
        int mini;

        for(int i=1;i<=40;i++)
        {
            ImageSet imgSet(QString("orl_faces"),i,10);
            imgSet.load();

            bool found = false;
            for(int k=0;k<10;k++)
            {
                imgSet.loadUp(k);
                if(i==idSetImgCautata && k==idImgCautata)
                {
                    memcpy(imagineCautata,imgSet.getImage(k)->getArray(),10304*sizeof(float));
                    QGraphicsScene* scene = new QGraphicsScene();
                    QGraphicsView* view = new QGraphicsView(scene);
                    QGraphicsPixmapItem* item = new QGraphicsPixmapItem(QPixmap::fromImage(imgSet.getQImage(k)));
                    scene->addItem(item);
                    ui->img2->setScene(scene);
                    found = true;
                }
            }
            if(found == true)
            {
                imgSet.freeUp(idImgCautata);
            }
            float *imaginiSet = imgSet.loadedToFloat();
            float rez, *result;

            result = (float*) malloc(imgSet.count()*sizeof(float));

            if(ui->norm1->isChecked())
            {
                cityblockNormAsync(imaginiSet,imagineCautata,10304, result, imgSet.count());
                rez = minimum(result,imgSet.count());
            }
            if(ui->norm2->isChecked())
            {
                euclideanNormAsync(imaginiSet,imagineCautata,10304, result, imgSet.count());
                rez = minimum(result,imgSet.count());
            }
            if(ui->norm3->isChecked())
            {
                cosNormAsync(imaginiSet,imagineCautata,10304, result, imgSet.count());
                rez = minimum(result,imgSet.count());
            }
            if(ui->norm4->isChecked())
            {
                cityblockNormAsync(imaginiSet,imagineCautata,10304, result, imgSet.count());
                rez = minimum(result,imgSet.count());
            }

            //qDebug()<<rez;
            if(i==1)
            {
                min = rez;
                mini = i;
            } else
            {
                if(rez<min)
                {
                    min = rez;
                    mini = i;
                }
            }

        }
        ImageSet imgSet(QString("orl_faces"),mini,10);
        imgSet.load();
        imgSet.loadUp(0);
        QGraphicsScene* scene = new QGraphicsScene();
        QGraphicsView* view = new QGraphicsView(scene);
        QGraphicsPixmapItem* item = new QGraphicsPixmapItem(QPixmap::fromImage(imgSet.getQImage(0)));
        scene->addItem(item);
        ui->img1->setScene(scene);
    }
    //KNN
    if(ui->knn->isChecked())
    {
        if(!(ui->norm1->isChecked() || ui->norm2->isChecked() || ui->norm3->isChecked() || ui->norm4->isChecked()))
            return;

        float min;
        int mini;

        for(int i=1;i<=40;i++)
        {
            ImageSet imgSet(QString("orl_faces"),i,10);
            imgSet.load();

            bool found = false;
            for(int k=0;k<10;k++)
            {
                imgSet.loadUp(k);
                if(i==idSetImgCautata && k==idImgCautata)
                {
                    memcpy(imagineCautata,imgSet.getImage(k)->getArray(),10304*sizeof(float));
                    QGraphicsScene* scene = new QGraphicsScene();
                    QGraphicsView* view = new QGraphicsView(scene);
                    QGraphicsPixmapItem* item = new QGraphicsPixmapItem(QPixmap::fromImage(imgSet.getQImage(k)));
                    scene->addItem(item);
                    ui->img2->setScene(scene);
                    found = true;
                }
            }
            if(found == true)
            {
                imgSet.freeUp(idImgCautata);
            }
            float *imaginiSet = imgSet.loadedToFloat();
            float rez, *result;

            result = (float*)malloc(imgSet.count()*sizeof(float));

            if(ui->norm1->isChecked())
            {
                cityblockNormAsync(imaginiSet,imagineCautata,10304, result, imgSet.count());
                rez = minimum(result,imgSet.count(), KVECINI);
            }
            if(ui->norm2->isChecked())
            {
                euclideanNormAsync(imaginiSet,imagineCautata,10304, result, imgSet.count());
                rez = minimum(result,imgSet.count(), KVECINI);
            }
            if(ui->norm3->isChecked())
            {
                cosNormAsync(imaginiSet,imagineCautata,10304, result, imgSet.count());
                rez = minimum(result,imgSet.count(), KVECINI);
            }
            if(ui->norm4->isChecked())
            {
                cityblockNormAsync(imaginiSet,imagineCautata,10304, result, imgSet.count());
                rez = minimum(result,imgSet.count(), KVECINI);
            }
            //qDebug()<<rez;
            if(i==1)
            {
                min = rez;
                mini = i;
            } else
            {
                if(rez<min)
                {
                    min = rez;
                    mini = i;
                }
            }

        }
        ImageSet imgSet(QString("orl_faces"),mini,10);
        imgSet.load();
        imgSet.loadUp(0);
        QGraphicsScene* scene = new QGraphicsScene();
        QGraphicsView* view = new QGraphicsView(scene);
        QGraphicsPixmapItem* item = new QGraphicsPixmapItem(QPixmap::fromImage(imgSet.getQImage(0)));
        scene->addItem(item);
        ui->img1->setScene(scene);
    }

    /*
    imagineCautata = (float*) malloc(10304*sizeof(float));
    float *imagineCautata2 = (float*) malloc(10304*sizeof(float));
    for(int i=0;i<10304;i++)
    {
        imagineCautata[i]=1;
        imagineCautata2[i]=1;
    }

    test1(imagineCautata,imagineCautata,10304);*/


    /*
    imgSet.loadUp(0);

    */
}
