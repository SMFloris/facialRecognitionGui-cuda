#-------------------------------------------------
#
# Project created by QtCreator 2015-10-11T15:50:45
#
#-------------------------------------------------

QT       += core gui widgets

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = facialRecognitionGui-cuda
TEMPLATE = app

SOURCES += main.cpp\
        mainwindow.cpp \
    imageset.cpp \
    image.cpp

HEADERS  += mainwindow.h \
    imageset.h \
    image.h

FORMS    += mainwindow.ui

# This makes the .cu files appear in your project
OTHER_FILES += cudaFuncs.cu
CUDA_SOURCES += cudaFuncs.cu

# project build directories

LIBS += -lcudart -lcuda

# C++ flags
QMAKE_CXXFLAGS_RELEASE = -O3

# Path to cuda toolkit install
CUDA_DIR = /usr/lib/nvidia-cuda-toolkit

# Path to header and libs files
INCLUDEPATH += $$CUDA_DIR/include
QMAKE_LIBDIR += /usr/lib/x86_64-linux-gnu     # Note I'm using a 64 bits Operating system

# GPU architecture
CUDA_ARCH = sm_12                       # I've a old device. Adjust with your compute capability

# Here are some NVCC flags I've always used by default.
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')
cuda.commands = nvcc -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}

cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -M ${QMAKE_FILE_NAME}
cuda.input = CUDA_SOURCES
cuda.output = ${QMAKE_FILE_BASE}_cuda.o

# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda
