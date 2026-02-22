cmake_minimum_required(VERSION 4.1)
project(MatrixCore LANGUAGES C CXX CUDA)

#set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1")

add_executable(MatrixCore main.cpp
        parser/token.h
        parser/tokenize.cpp
        parser/parser.h
        parser/parser-utils.h
        parser/error.h
        parser/node.h
        parser/parsing.cpp
        matrix.h
        device/sub_matrix.cu
        parser/token.cpp
        parser/node.cpp
        parser/error.cpp
        parser/interpreter.cpp
        parser/bytecode.h
        parser/bytecode.cpp
        parser/optimizer.cpp
        device/device_submatrix.cuh
        device/device_constructor.cuh
        device/device_instruction.cuh
        device/cfc_solver.cu
        device/device_bytecode.cuh
        device/subqubo_merge_tree_dp.cu
        device/subqubo_merge_tree_dp.cu
)
