#include "mainwindow.h"
#include "imageset.h"
#include <QApplication>
#include <math.h>
#include <stdio.h>

//a,b vectori, c destinatie, n dimensiune
extern void proprietati();

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;

    w.show();
    //proprietati();


    return a.exec();
}
