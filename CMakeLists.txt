cmake_minimum_required(VERSION 4.1)
project(MatrixCore LANGUAGES C CXX CUDA)

#set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1")

add_executable(MatrixCore main.cpp
        token.h
        tokenize.cpp
        parser.h
        parser-utils.h
        error.h
        node.h
        parsing.cpp
        matrix.h
        sub_matrix.cu
#        sub_matrix.cuh
#        sub_matrix.cuh
        parser-utils.cpp
        token.cpp
        node.cpp
        error.cpp
)
