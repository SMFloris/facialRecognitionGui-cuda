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

extern void cumSum(float *out, float *in, unsigned int n);
extern void test1(float *a, float *b, unsigned long n);

extern float euclideanNormAsync(float *a, float *b, unsigned long n, unsigned long nrPoze);


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
    int idImgCautata = 9;
    float *imagineCautata;

    QGraphicsScene* scene = new QGraphicsScene();
    QGraphicsView* view = new QGraphicsView(scene);

    for(int i=1;i<=40;i++)
    {
        ImageSet imgSet(QString("orl_faces"),i,10);
        imgSet.load();
        for(int k=0;k<10;k++)
        {
            imgSet.loadUp(k);
        }
        if(i*10-idImgCautata>=0 && i*10-idImgCautata<10)
        {
            imagineCautata = imgSet.getImage(i*10-idImgCautata)->getArray();

            QGraphicsPixmapItem* item = new QGraphicsPixmapItem(QPixmap::fromImage(imgSet.getQImage(i*10-idImgCautata)));
            scene->addItem(item);
            ui->img2->setScene(scene);
            imgSet.freeUp(i*10-idImgCautata);
        }
        float *imaginiSet = imgSet.loadedToFloat();
        float a = euclideanNormAsync(imaginiSet,imagineCautata,10304,imgSet.count());
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
