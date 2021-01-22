#include "MatrixKernel.cuh"


/////////////////////////////////////////////////////////////////
// Konstruktory
/////////////////////////////////////////////////////////////////
template<typename T>
MatrixKernel<T>::MatrixKernel(const T * h_v1, const T * h_v2, T * h_v3, const int r, const int c){
    rows = r;
    cols = c;
    size = r*c;

    cudaMalloc((void **)&d_v1, size * sizeof(T));
    cudaMalloc((void **)&d_v2, size * sizeof(T));
    cudaMalloc((void **)&d_v3, size * sizeof(T));

    cudaMemcpy(d_v1, h_v1, size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2, h_v2, size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v3, h_v3, size * sizeof(T), cudaMemcpyHostToDevice);
}


template<typename T>
MatrixKernel<T>::MatrixKernel(const T* h_v1, const T h_s, T* h_v3, const int r, const int c){
    rows = r;
    cols = c;
    size = r*c;

    cudaMalloc((void **)&d_v1, size * sizeof(T));
    cudaMalloc((void **)&d_s,         sizeof(T));
    cudaMalloc((void **)&d_v3, size * sizeof(T));

    cudaMemcpy(d_v1, h_v1, size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s,  &h_s,        sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v3, h_v3, size * sizeof(T), cudaMemcpyHostToDevice);
}


template<typename T>
MatrixKernel<T>::MatrixKernel(const T * h_v1, T * h_v3, const int r, const int c){
    rows = r;
    cols = c;
    size = r*c;

    cudaMalloc((void **)&d_v1, size * sizeof(T));
    cudaMalloc((void **)&d_v3, size * sizeof(T));

    cudaMemcpy(d_v1, h_v1, size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v3, h_v3, size * sizeof(T), cudaMemcpyHostToDevice);
}


/////////////////////////////////////////////////////////////////
// Destruktory
/////////////////////////////////////////////////////////////////
template<typename T>
MatrixKernel<T>::~MatrixKernel(){
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_v3);
    cudaFree(d_s);
}


/////////////////////////////////////////////////////////////////
// Kernel - dodawania macierzy
/////////////////////////////////////////////////////////////////
template<typename T>
__global__ void addmKernel(const T* d_v1, const T* d_v2, T* d_v3, size_t size){
    int index = blockIdx.x * 1024 + threadIdx.y * 32 + threadIdx.x;
    if(index < size){
        d_v3[index] = d_v1[index] + d_v2[index];
    }
}

template<typename T>
__host__ void MatrixKernel<T>::addmMatrixKernel(){
    int dbx = 32;
    int dby = 32;
    int dgx = static_cast<int>(this->cols/dbx)+1;
    int dgy = static_cast<int>(this->rows/dby)+1;

    dim3 dimBlock(dbx, dby);
    dim3 dimGrid (dgx, dgy);
    addmKernel<<<dimGrid, dimBlock>>>(this->d_v1, this->d_v2, this->d_v3, this->size);
}


/////////////////////////////////////////////////////////////////
// Kernel - odejmowania macierzy
/////////////////////////////////////////////////////////////////
template<typename T>
__global__ void submKernel(const T* d_v1, const T* d_v2, T* d_v3, size_t size){
    int index = blockIdx.x * 1024 + threadIdx.y * 32 + threadIdx.x;
    if(index < size){
        d_v3[index] = d_v1[index] - d_v2[index];
    }
}


template<typename T>
__host__ void MatrixKernel<T>::submMatrixKernel(){
    int dbx = 32;
    int dby = 32;
    int dgx = static_cast<int>(this->cols/dbx)+1;
    int dgy = static_cast<int>(this->rows/dby)+1;

    dim3 dimBlock(dbx, dby);
    dim3 dimGrid (dgx, dgy);
    submKernel<<<dimGrid, dimBlock>>>(this->d_v1, this->d_v2, this->d_v3, this->size);
}

/////////////////////////////////////////////////////////////////
// funkcja mnozenia macierzy z pamiecia shared
/////////////////////////////////////////////////////////////////

// template<typename T>
// __global__ void mulmKernelSHARED(const T* d_v1, const T* d_v2, T* d_v3, const int r, const int c){
    
//     const int BLOCK_SIZE = 32;

//     int blockRow = blockIdx.y;
//     int blockCol = blockIdx.x;

//     T * Csub = &d_v3[r * BLOCK_SIZE * blockRow + BLOCK_SIZE * blockCol];
    
//     double Cvalue = 0;

//     int row = threadIdx.y;
//     int col = threadIdx.x;

//     int n = static_cast<int>(r/BLOCK_SIZE)+1;
//     for(int i=0; i<n; i++){
        
//         T * Asub = &d_v1[r * BLOCK_SIZE * blockRow + BLOCK_SIZE * i];
//         T * Bsub = &d_v2[r * BLOCK_SIZE * i + BLOCK_SIZE * blockCol];

//         __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
//         As[row][col] = Asub[row*r + col];

//         __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];
//         Bs[row][col] = Bsub[row*r + col];

//         __syncthreads();

//         for(int j=0; j < BLOCK_SIZE; j++){
//             Cvalue += As[row][j] * Bs[j][col];
//         }

//         __syncthreads();

//         Csub[row*r + col] = Cvalue;
//     }
// }

/////////////////////////////////////////////////////////////////
// Kernel - mnozenia macierzy
/////////////////////////////////////////////////////////////////
template<typename T>
__global__ void mulmKernel(const T* d_v1, const T* d_v2, T* d_v3, const int r, const int c){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < r && col < c){
        for(int i=0; i<r; ++i){
            d_v3[row * r + col] += d_v1[row * r + i] * d_v2[i * r + col];
        }
    }
}


template<typename T>
__host__ void MatrixKernel<T>::mulmMatrixKernel(){
    int dbx = 32;
    int dby = 32;
    int dgx = static_cast<int>(this->cols/dbx)+1;
    int dgy = static_cast<int>(this->rows/dby)+1;    

    dim3 dimBlock(dbx, dby);
    dim3 dimGrid (dgx, dgy);
    mulmKernel<<<dimGrid, dimBlock>>>(this->d_v1, this->d_v2, this->d_v3, this->rows, this->cols);
    cudaDeviceSynchronize();
}


/////////////////////////////////////////////////////////////////
// Kernel - dodawania stałej
/////////////////////////////////////////////////////////////////
template<typename T>
__global__ void addsKernel(const T* d_v1, const T* d_s, T* d_v3, size_t size){
    int index = blockIdx.x * 1024 + threadIdx.y * 32 + threadIdx.x;
    if(index < size){
        d_v3[index] = d_v1[index] + *d_s;
    }
}


template<typename T>
__host__ void MatrixKernel<T>::addsMatrixKernel(){
    int dbx = 32;
    int dby = 32;
    int dgx = static_cast<int>(this->cols/dbx)+1;
    int dgy = static_cast<int>(this->rows/dby)+1;

    dim3 dimBlock(dbx, dby);
    dim3 dimGrid (dgx, dgy);
    addsKernel<<<dimGrid, dimBlock>>>(this->d_v1, this->d_s, this->d_v3, this->size);
}


/////////////////////////////////////////////////////////////////
// Kernel - odejmowania stalej
/////////////////////////////////////////////////////////////////
template<typename T>
__global__ void subsKernel(const T* d_v1, const T* d_s, T* d_v3, size_t size){
    int index = blockIdx.x * 1024 + threadIdx.y * 32 + threadIdx.x;
    if(index < size){
        d_v3[index] = d_v1[index] - *d_s;
    }
}


template<typename T>
__host__ void MatrixKernel<T>::subsMatrixKernel(){
    int dbx = 32;
    int dby = 32;
    int dgx = static_cast<int>(this->cols/dbx)+1;
    int dgy = static_cast<int>(this->rows/dby)+1;

    dim3 dimBlock(dbx, dby);
    dim3 dimGrid (dgx, dgy);
    subsKernel<<<dimGrid, dimBlock>>>(this->d_v1, this->d_s, this->d_v3, this->size);
}


/////////////////////////////////////////////////////////////////
// Kernel - mnozenia stalej
/////////////////////////////////////////////////////////////////
template<typename T>
__global__ void mulsKernel(const T* d_v1, const T* d_s, T* d_v3, size_t size){
    int index = blockIdx.x * 1024 + threadIdx.y * 32 + threadIdx.x;
    if(index < size){
        d_v3[index] = d_v1[index] * *d_s;
    }
}


template<typename T>
__host__ void MatrixKernel<T>::mulsMatrixKernel(){
    int dbx = 32;
    int dby = 32;
    int dgx = static_cast<int>(this->cols/dbx)+1;
    int dgy = static_cast<int>(this->rows/dby)+1;    

    dim3 dimBlock(dbx, dby);
    dim3 dimGrid (dgx, dgy);
    mulsKernel<<<dimGrid, dimBlock>>>(this->d_v1, this->d_s, this->d_v3, this->size);
    cudaDeviceSynchronize();
}


/////////////////////////////////////////////////////////////////
// Kernel - transpozycji
/////////////////////////////////////////////////////////////////
template<typename T>
__global__ void transposeKernel(const T* d_v1, T* d_v3, const int r, const int c){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < r && col < c){
        d_v3[col * r + row] = d_v1[row * r + col];
    }
}


template<typename T>
__host__ void MatrixKernel<T>::transposeMatrixKernel(){
    int dbx = 32;
    int dby = 32;
    int dgx = static_cast<int>(this->cols/dbx)+1;
    int dgy = static_cast<int>(this->rows/dby)+1;    

    dim3 dimBlock(dbx, dby);
    dim3 dimGrid (dgx, dgy);
    transposeKernel<<<dimGrid, dimBlock>>>(this->d_v1, this->d_v3, this->rows, this->cols);
    cudaDeviceSynchronize();
}


/////////////////////////////////////////////////////////////////
// przypisanie wyniku do hosta
/////////////////////////////////////////////////////////////////
template<typename T>
void MatrixKernel<T>::get_d_v3(T * h_v3){
    cudaMemcpy(h_v3, d_v3, this->size * sizeof(T), cudaMemcpyDeviceToHost);
}


