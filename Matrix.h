#ifndef __Matrix_H
#define __Matrix_H

#include <iostream>
#include <iomanip>
#include <vector>

template <typename T>
class Matrix{
    private: 
        std::vector<T> mat;
        int rows;
        int cols;
        bool gpu;

    public:
        // konstruktory
        Matrix();
        Matrix(const int r, const int c, const T& s);
        Matrix(const Matrix<T>& m);

        // destruktor
        virtual ~Matrix();

        // operator przypisania
        Matrix<T>& operator= (const Matrix<T>& m);

        // operacje macierzowe
        auto  operator+ (const Matrix<T>& m);
        auto  operator- (const Matrix<T>& m);
        auto  operator* (const Matrix<T>& m);
        auto& operator+=(const Matrix<T>& m);
        auto& operator-=(const Matrix<T>& m);
        auto& operator*=(const Matrix<T>& m);
        Matrix<T> transpose();

        // operacje skalarne
               auto operator+ (const T& s) const;
               auto operator- (const T& s) const;
               auto operator* (const T& s) const;
        friend auto operator+ (const T& s, const Matrix<T>& m){ return m + s; }
        friend auto operator- (const T& s, const Matrix<T>& m){ return m*(-1) + s; }
        friend auto operator* (const T& s, const Matrix<T>& m){ return m * s; }

        // pobranie elementu z danej pozycji        
              T& operator()(const int& r, const int& c);
        const T& operator()(const int& r, const int& c) const;
        
        // gettery
        int get_row() const;
        int get_col() const;
        std::vector<T> get_mat() const;

        // settery
        void set_element_at(const int& r, const int& c, const T& s);
        void set_gpu(const bool b);

        // wyswietlanie
        std::ostream& print(std::ostream& os) const;
        friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& m){
            return m.print(os);
        }
};

#include "Matrix.cpp"

#endif
