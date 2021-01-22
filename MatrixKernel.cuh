#ifndef __MatrixKernel_H
#define __MatrixKernel_H

#include <cuda.h>
#include <cuda_runtime.h>

template<typename T>
class MatrixKernel{
    private:
        T* d_v1;
        T* d_v2;
        T* d_v3;
        T* d_s;
        int rows;
        int cols;
        size_t size;

    public:
        // konstruktory
        MatrixKernel(const T * h_v1, const T * h_v2, T * h_v3, const int r, const int c);
        MatrixKernel(const T * h_v1, const T   h_s,  T * h_v3, const int r, const int c);
        MatrixKernel(const T * h_v1,                 T * h_v3, const int r, const int c);

        // destruktor
        virtual ~MatrixKernel();

        // zwracanie wynikowego vectora do hosta
        void get_d_v3(T * h_v3);

        // funkcje obsługujace kernele do obliczeń macierzowych
        void addmMatrixKernel();
        void submMatrixKernel();
        void mulmMatrixKernel();
        void transposeMatrixKernel();

        // funkcje obsługujace kernele do obliczeń skalarnych
        void addsMatrixKernel();
        void subsMatrixKernel();
        void mulsMatrixKernel();
};

#include "MatrixKernel.cu"

#endif