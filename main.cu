#include "Matrix.h"
#include <iostream>
#include <complex>
#include <chrono>
#include <random>
#include <fstream>
#include <omp.h>

using Time = std::chrono::high_resolution_clock;
using fsec = std::chrono::duration<float>;

void gen_random(Matrix<int> & m,    const int min, const int max);
void gen_random(Matrix<double> & m, const int min, const int max);


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

int main(){

    omp_set_num_threads(omp_get_thread_num());
  
    // rozmiar macierzy
    const int roz_M = 3;

    Matrix<double> A (roz_M, roz_M, 0.0);   gen_random(A, -5, 5);
    Matrix<double> B (roz_M, roz_M, 0.0);   gen_random(B, -5, 5);

    // Wynik macierzy na CPU i zmierzenie czasu
    auto start_CPU = Time::now();
    Matrix<double> C = A*B + 6*A.transpose() + A - 3*B;
    auto end_CPU = Time::now();

    A.set_gpu(true);
    B.set_gpu(true);

    // Wynik macierzy na GPU i zmierzenie czasu
    auto start_GPU = Time::now();
    Matrix<double> D = A*B + 6*A.transpose() + A - 3*B;
    auto end_GPU = Time::now();

    fsec fs1 = end_CPU - start_CPU;
    fsec fs2 = end_GPU - start_GPU;

    std::cout << " * MACIERZ A *" << std::endl;
    std::cout << A ;
    std::cout << " * MACIERZ B *" << std::endl;
    std::cout << B ;
    std::cout << " * MACIERZ C *" << std::endl;
    std::cout << C ;
    std::cout << " * MACIERZ D *" << std::endl;
    std::cout << D ;
    std::cout << "[CPU] Czas macierzy C: " << std::setprecision(16) << fs1.count() << "s" << std::endl;
    std::cout << "[GPU] Czas macierzy D: " << std::setprecision(16) << fs2.count() << "s" << std::endl;

    return 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void gen_random(Matrix<int> & m, const int min, const int max){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distrib(min, max);

    int r = m.get_row();
    int c = m.get_col();

    for(int i=0; i<r; i++){
        for(int j=0; j<c; j++){
            m.set_element_at(i,j, distrib(gen));
        }
    }
}

void gen_random(Matrix<double> & m, const int min, const int max){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrib(min, max);

    int r = m.get_row();
    int c = m.get_col();

    for(int i=0; i<r; i++){
        for(int j=0; j<c; j++){
            m.set_element_at(i,j, distrib(gen));
        }
    }
}