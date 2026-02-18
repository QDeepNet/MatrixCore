#ifndef MATRIXCORE_SUB_MATRIX_CUH
#define MATRIXCORE_SUB_MATRIX_CUH

#include <stdint.h>
#include <stdlib.h>
#include "device_matrix.cuh"
#include "device_constructor.cuh"
#include "device_instruction.cuh"









__global__  void __constructor_interpreter(const device_constructor_t *c, const device_matrix_t *m, const device_instruction_t *i);
static void constructor_interpreter(const device_constructor_t *c, const device_matrix_t *m, const device_instruction_t *i) {
    __constructor_interpreter<<<1024, 1024>>>(c, m, i);
}


#endif //MATRIXCORE_SUB_MATRIX_CUH