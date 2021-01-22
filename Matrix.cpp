#include "Matrix.h"
#include "MatrixKernel.cuh"

/////////////////////////////////////////////////////////////////
// Konstruktory
/////////////////////////////////////////////////////////////////
template<typename T>
Matrix<T>::Matrix(){
    mat.resize(1, 0);
    rows = 1;
    cols = 1;
    gpu = false;
}


template<typename T>
Matrix<T>::Matrix(const int r, const int c, const T& s){
    mat.resize(r*c, s);
    rows = r;
    cols = c;
    gpu = false;
}


template<typename T>
Matrix<T>::Matrix(const Matrix<T>& m){
    mat  = m.get_mat();
    rows = m.get_row();
    cols = m.get_col();
    gpu = false;
}


/////////////////////////////////////////////////////////////////
// Destruktory
/////////////////////////////////////////////////////////////////
template<typename T>
Matrix<T>::~Matrix(){
    //std::cout << "Macierz usunieta" << std::endl;
}


/////////////////////////////////////////////////////////////////
// operator przypisania
/////////////////////////////////////////////////////////////////
template<typename T>
Matrix<T>&  Matrix<T>::operator= (const Matrix<T>& m){
    if(this == &m){
        return *this;
    }

    rows = m.get_row();
    cols = m.get_col();

    mat.resize(rows*cols);

    int i, j;
    #pragma omp parallel for private(i,j)
    for (i=0; i<this->rows; i++) {
        for (j=0; j<this->cols; j++) {
            mat[i*this->rows + j] = m(i, j);
        }
    }
    
    return *this;
}


/////////////////////////////////////////////////////////////////
// operacje macierzowe
/////////////////////////////////////////////////////////////////
template<typename T>
auto Matrix<T>::operator+ (const Matrix<T>& m){
    Matrix<T> sum(this->rows, this->cols, 0.0);
    
    if(gpu == true){
        MatrixKernel<T> MX(this->mat.data(), m.get_mat().data(), sum.get_mat().data(), this->rows, this->cols);
        MX.addmMatrixKernel();
        MX.get_d_v3(sum.mat.data());
    } 
    else if(gpu == false){
        int i, j;
        #pragma omp parallel for private(i,j)
        for (i=0; i<this->rows; i++){
            for (j=0; j<this->cols; j++){
                sum(i, j) = this->mat[i*this->rows + j] + m(i, j);
            }
        }
    }
    
    return sum;
}


template<typename T>
auto Matrix<T>::operator- (const Matrix<T>& m){
    Matrix<T> sub(this->rows, this->cols, 0.0);

    if(gpu == true){
        MatrixKernel<T> MX(this->mat.data(), m.get_mat().data(), sub.get_mat().data(), this->rows, this->cols);
        MX.submMatrixKernel();
        MX.get_d_v3(sub.mat.data());
    }
    else if(gpu == false){
        int i, j;
        #pragma omp parallel for private(i,j)
        for (i=0; i<this->rows; i++){
            for (j=0; j<this->cols; j++){
                sub(i, j) = this->mat[i*this->rows + j] - m(i, j);
            }
        }
    }
    
    return sub;
}


template<typename T>
auto Matrix<T>::operator* (const Matrix<T>& m){
    Matrix<T> product(this->rows, this->cols, 0.0);

    if(gpu == true){
        MatrixKernel<T> MX(this->mat.data(), m.get_mat().data(), product.get_mat().data(), this->rows, this->cols);
        MX.mulmMatrixKernel();
        MX.get_d_v3(product.mat.data());
    } 
    else if(gpu == false){
        int i, j, k;
        T c = 0;
        #pragma omp parallel for reduction(+:c) private(i,j)
        for (i=0; i<this->rows; i++) {
            for (j=0; j<this->cols; j++) {
                c = 0;
                for (k=0; k<this->rows; k++) {
                    c += this->mat[i*this->rows + k] * m(k,j);
                }
                product(i,j) = c;
            }
        }
    }
    
    return product;
}


template<typename T>
auto& Matrix<T>::operator+=(const Matrix<T>& m){
    Matrix<T> sub = (*this) + m;
    (*this) = sub;
    return *this;
}


template<typename T>
auto& Matrix<T>::operator-=(const Matrix<T>& m){
    Matrix<T> sub = (*this) - m;
    (*this) = sub;
    return *this;
}


template<typename T>
auto& Matrix<T>::operator*=(const Matrix<T>& m){
    Matrix<T> product = (*this) * m;
    (*this) = product;
    return *this;
}


template<typename T>
Matrix<T> Matrix<T>::transpose(){
    Matrix<T> trans(this->rows, this->cols, 0.0);
    if(gpu == true){
        MatrixKernel<T> MX(this->mat.data(), trans.get_mat().data(), this->rows, this->cols);
        MX.transposeMatrixKernel();
        MX.get_d_v3(trans.mat.data());
    } 
    else if(gpu == false){
        int i, j;
        #pragma omp parallel for private(i,j)
        for (i=0; i<this->rows; i++){
            for (j=0; j<this->cols; j++){
                trans(i, j) = this->mat[j*this->rows + i];
            }
        }
    }
    
    return trans;
}


/////////////////////////////////////////////////////////////////
// operacje skalarne
/////////////////////////////////////////////////////////////////
template<typename T>
auto Matrix<T>::operator+(const T& s) const {
    Matrix<T> sum(this->rows, this->cols, 0.0);

    if(gpu == true){
        MatrixKernel<T> MX(this->mat.data(), s, sum.get_mat().data(), this->rows, this->cols);
        MX.addsMatrixKernel();
        MX.get_d_v3(sum.mat.data());
    }
    else if(gpu == false){
        int i, j;
        #pragma omp parallel for private(i,j)
        for (i=0; i<this->rows; i++) {
            for (j=0; j<this->cols; j++) {
                sum(i,j) = this->mat[i*this->rows + j] + s;
            }
        }
    }

    return sum;
}


template<typename T>
auto Matrix<T>::operator-(const T& s) const {
    Matrix<T> sub(this->rows, this->cols, 0.0);

    if(gpu == true){
        MatrixKernel<T> MX(this->mat.data(), s, sub.get_mat().data(), this->rows, this->cols);
        MX.subsMatrixKernel();
        MX.get_d_v3(sub.mat.data());
    } 
    else if(gpu == false){
        int i, j;
        #pragma omp parallel for private(i,j)
        for (i=0; i<this->rows; i++) {
            for (j=0; j<this->cols; j++) {
                sub(i,j) = this->mat[i*this->rows + j] - s;
            }
        }
    }

    return sub;
}


template<typename T>
auto Matrix<T>::operator*(const T& s) const {
    Matrix<T> product(this->rows, this->cols, 0.0);

    if(gpu == true){
        MatrixKernel<T> MX(this->mat.data(), s, product.get_mat().data(), this->rows, this->cols);
        MX.mulsMatrixKernel();
        MX.get_d_v3(product.mat.data());
    } 
    else if(gpu == false){
        int i, j;
        #pragma omp parallel for private(i,j)
        for (i=0; i<this->rows; i++) {
            for (j=0; j<this->cols; j++) {
                product(i,j) = this->mat[i*this->rows + j] * s;
            }
        }
    }

    return product;
}


/////////////////////////////////////////////////////////////////
// pobranie elementu z danej pozycji 
/////////////////////////////////////////////////////////////////
template<typename T>
T& Matrix<T>::operator()(const int& r, const int& c){
    return this->mat[r*this->rows + c];
}


template<typename T>
const T& Matrix<T>::operator()(const int& r, const int& c) const{
    return this->mat[r*this->rows + c];
}



/////////////////////////////////////////////////////////////////
// gettery
/////////////////////////////////////////////////////////////////
template<typename T>
int Matrix<T>::get_row() const{
    return this->rows;
}


template<typename T>
int Matrix<T>::get_col() const{
    return this->cols;
}


template<typename T>
std::vector<T> Matrix<T>::get_mat() const{
    return this->mat;
}


/////////////////////////////////////////////////////////////////
// settery
////////////////////////////////////////////////////////////////
template<typename T>
void Matrix<T>::set_element_at(const int& r, const int& c, const T& s){
    this->mat[r*this->rows + c] = s;
}


template<typename T>
void Matrix<T>::set_gpu(const bool b){
    gpu = b;
}


/////////////////////////////////////////////////////////////////
// wyswietlanie
/////////////////////////////////////////////////////////////////
template<typename T>
std::ostream& Matrix<T>::print(std::ostream& os) const
{
    std::string l(this->cols*17, '-');
    std::cout << l << std::endl;
    for (int i=0; i<this->rows; i++) {
        std::cout << "|";
        for (int j=0; j<this->cols; j++) {
            os << std::setprecision(2) << std::fixed << std::setw(16) << this->mat[i*this->rows + j];
        }
        std::cout << " |" << std::endl;
        std::cout << "|  " << std::setw(this->cols*16) << "|" << std::endl; 
    }
    std::cout << l << std::endl;


    return os;
}