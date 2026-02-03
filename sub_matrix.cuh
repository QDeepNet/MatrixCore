#ifndef MATRIXCORE_SUB_MATRIX_CUH
#define MATRIXCORE_SUB_MATRIX_CUH

#include <stdint.h>
#include <stdlib.h>


typedef struct {
    int64_t n;
    int64_t offset_i;
    int64_t offset_j;
} device_data_t;


typedef struct {
    device_data_t *data;

    int *top; // For Stack
    int64_t *stack;

    int64_t *matrix; // matrix n by n
    int64_t *ids_i;
    int64_t *ids_j;
    int64_t *q_j;
} device_matrix_t;

typedef uint8_t device_instruction;
typedef struct {
    int64_t min_i;
    int64_t max_i;

    int64_t min_j;
    int64_t max_j;
} instruction_params_t;

__global__  void __constructor_interpreter(device_matrix_t *m, instruction_params_t *p, device_instruction *list, uint64_t size);

static __inline__ void device_matrix_init(uint64_t n) {
    device_matrix_t *m = (device_matrix_t *) malloc(sizeof(device_matrix_t));
    cudaMalloc(&m->data, sizeof(device_data_t));

    cudaMalloc(&m->top, sizeof(uint8_t));
    cudaMalloc(&m->stack, 1024 * 1024 * 32 * sizeof(int64_t));

    cudaMalloc(&m->matrix, n * n * sizeof(int64_t));
    cudaMalloc(&m->ids_i, n * sizeof(int64_t));
    cudaMalloc(&m->ids_j, 1024 * sizeof(int64_t));
    cudaMalloc(&m->q_j, 1024 * sizeof(int64_t));

}
static __inline__ void device_matrix_free(device_matrix_t *m) {
    cudaFree(&m->data);
    cudaFree(&m->top);
    cudaFree(&m->stack);
    cudaFree(&m->matrix);
    cudaFree(&m->ids_i);
    cudaFree(&m->ids_j);
    cudaFree(&m->q_j);
}






static void constructor_interpreter(device_matrix_t *m, instruction_params_t *p, device_instruction *list, const uint64_t size) {
    __constructor_interpreter<<<1024, 1024>>>(m, p, list, size);
}


#endif //MATRIXCORE_SUB_MATRIX_CUH