cmake_minimum_required(VERSION 4.1)
project(MatrixCore)

#set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)


add_executable(MatrixCore main.cpp
        token.h
        tokenize.c
        parser.h
        parser-utils.h
        error.h
        node.h
        parsing.c
        matrix.h
)
