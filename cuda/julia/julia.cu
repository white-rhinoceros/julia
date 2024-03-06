
#include <iostream>
//#include "complex.h"

using namespace std;

extern "C" {
     __global__ void dot__(float *dev) {

     }


    void dot(float *source, float *dev, size_t count) {
        cudaMemcpy(dev, source, count * sizeof(float), cudaMemcpyHostToDevice);

        dot__<<<100, 100>>>(dev);

        cudaFree(dev);
    }

    void dot2() {
        cout << "Calling gpu dot product" << endl;
    }
}

///**
// * Основная функция вызываемая на GPU.
// */
//__global__ void kernel(unsigned char * bitmap) {
//    int x = blockIdx.x;
//    int y = blockIdx.y;
//    int offset = x + y*gridDim.x;
//}
//
//__device__ int julia(int x, int y) {
//
//}
//
//extern "C" {
///**
// * Точка входа CPU кода
// *
// * @param dim_x Размер изображения по x.
// * @param dim_y Размер изображения по y.
// * @param bitmap Указатель на массив точек изображения выделенный в памяти GPU.
// * @return Не нулевое значение свидетельствует об ошибке.
// */
//int fill(size_t dim_x, size_t dim_y, unsigned char * bitmap) {
//    dim3 grid(dim_x, dim_y);
//    kernel<<<grid, 1>>>(bitmap);
//
//    return 0;
//}
//}

