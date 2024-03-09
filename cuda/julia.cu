#include "./include/complex.h"

extern "C" {
    void julia(unsigned char *pixels, size_t bound_x, size_t bound_y);
}

// __global__ void kernel(unsigned char * ptr, unsigned int bound_x, unsigned int bound_y);
// __device__ unsigned int calc_val(unsigned int x, unsigned int y, unsigned int bound_x, unsigned int bound_y);

__device__ unsigned int calc_val(unsigned int x, unsigned int y, unsigned int bound_x, unsigned int bound_y) {
    const float scale = 1.5;

    const float max_x2 = (float)bound_x/2;
    float jx = scale * (max_x2 - (float)x)/max_x2;

    const float max_y2 = (float)bound_y/2;
    float jy = scale * (max_y2 - (float)y)/max_y2;

    Complex c(-0.8, 0.156);
    Complex a(jx, jy);

    for (int i = 0; i < 200; i++) {
        a = a*a + c;

        if (a.magnitude2() > 1000) {
            return 0;
        }
    }

    return 1;
}

/**
 * Основная функция вызываемая на GPU.
 */
__global__ void kernel(unsigned char * ptr, unsigned int bound_x, unsigned int bound_y) {
    // Поучить позицию пикселя.
    unsigned int x = blockIdx.x;
    unsigned int y = blockIdx.y;
    unsigned int offset = x + y*gridDim.x;

    // Вычислить значение в этой позиции.
    unsigned int val = calc_val(x, y, bound_x, bound_y);

    ptr[offset*4 + 0] = 255 * val;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}

/**
 * Точка входа, вызывается из Rust кода.
 *
 * @param pixels Указатель на массив точек изображения выделенный в памяти GPU.
 * @param bound_x Размер изображения по x.
 * @param bound_y Размер изображения по y.
 */
void julia(unsigned char *pixels, size_t bound_x, size_t bound_y)
{
    // Используем только сетку (блоки в сетке).
    dim3 grid(bound_x, bound_y, 1);
    kernel<<<grid, 1>>>(pixels, bound_x, bound_y);
}


// Полезности...
// #include <iostream>
//::printf("%d\n", val);
// cudaMemcpy(res, dev_res, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
