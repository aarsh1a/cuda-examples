cmake_minimum_required(VERSION 3.8)
project(CudaExamples LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)

include_directories(include)

set(SOURCES
    src/vector_addition.cu
    src/matrix_multiplication.cu
    src/image_processing.cu
)

add_executable(cuda_examples ${SOURCES})