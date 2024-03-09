cmake_minimum_required(VERSION 3.27)
project(julia CUDA)

set(CMAKE_CXX_STANDARD 11)

include_directories(include)

add_library(julia STATIC julia.cu)
add_library(sph STATIC sph.cu)

set_target_properties(julia PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)