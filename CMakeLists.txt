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
#        sub_matrix.cuh
#        sub_matrix.cuh
        parser/parser-utils.cpp
        parser/token.cpp
        parser/node.cpp
        parser/error.cpp
        parser/interpreter.cpp
        parser/bytecode.h
        parser/bytecode.cpp
        parser/optimizer.cpp
        solver/cfc.h
        solver/adapter.h
        solver/cpu_adapter.cpp
        device/device_matrix.cuh
        solver/gpu_adapter.cu
        device/sub_matrix.cuh
        device/device_constructor.cuh
        device/device_instruction.cuh
        device/ising_converter.cuh
        device/ising_converter.cu
)

# 1. Function for computation the total energy of the vector
# 2. Understand the all computations in the Ipynb

# 3. Make the graph like construction
