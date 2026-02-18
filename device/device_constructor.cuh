#ifndef MATRIXCORE_DEVICE_CONSTRUCTOR_CUH
#define MATRIXCORE_DEVICE_CONSTRUCTOR_CUH

#include <stdint.h>
#include <stdlib.h>

typedef struct {

    int *stack_top; // For Stack
    int64_t *stack;

    int32_t *offset_j;
    int64_t *ids_j; // sub qubo indexes
    int64_t *q_j; // spins of the j variable (1 if included in subqubo)
} device_constructor_t;


static __inline__ device_constructor_t *device_constructor_init() {
    const auto c = static_cast<device_constructor_t *>(malloc(sizeof(device_constructor_t)));
    cudaMalloc(&c->stack_top, 1024 * sizeof(uint8_t));
    cudaMalloc(&c->stack, 1024 * 1024 * 32 * sizeof(int64_t));

    cudaMalloc(&c->offset_j, sizeof(int32_t));
    cudaMalloc(&c->ids_j, 1024 * sizeof(int64_t));
    cudaMalloc(&c->q_j, 1024 * sizeof(int64_t));

    return c;
}
static __inline__ void device_constructor_free(device_constructor_t *m) {
    cudaFree(&m->stack_top);
    cudaFree(&m->stack);

    cudaFree(&m->offset_j);
    cudaFree(&m->ids_j);
    cudaFree(&m->q_j);
}

#endif //MATRIXCORE_DEVICE_CONSTRUCTOR_CUH