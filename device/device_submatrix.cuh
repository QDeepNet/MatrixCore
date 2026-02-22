#ifndef MATRIXCORE_DEVICE_MATRIX_CUH
#define MATRIXCORE_DEVICE_MATRIX_CUH

#include <stdint.h>
#include <stdlib.h>

#define MATRIX_MAX_N 1024
#define MATRIX_MAX_V 16
#define MATRIX_MAX_D 16

#define MATRIX_STACK 32


typedef struct {
    int32_t n;
    int32_t v;      // Number of variations of input data
    int32_t d;      // Number of solution from the Solver
    int64_t *ids_i; // indexes to make sub qubo

    float_t *J;     // n x n (row-major) Ising
    float_t *h;     // n     optional

    // ---- CFC state/output on device ----
    float_t *e;     // d x v energy
    int8_t  *spin;  // n x d x v optional, size = n, writes +/-1 if non-null

    // ---- solver params ----
    int32_t  n_iter;  // iterations
    float_t  dt;      // typically 0.1f
    float_t  clamp;   // typically 1.5f
    uint32_t seed;    // for optional init
} device_submatrix_t;


static __inline__ device_submatrix_t *device_submatrix_init(const uint64_t n, const uint64_t d, const uint64_t v) {
    const auto m = static_cast<device_submatrix_t *>(malloc(sizeof(device_submatrix_t)));

    cudaMalloc(&m->ids_i, MATRIX_MAX_N * sizeof(int64_t));

    cudaMalloc(&m->J,     MATRIX_MAX_N * MATRIX_MAX_N * sizeof(float_t));
    cudaMalloc(&m->h,     MATRIX_MAX_N * sizeof(float_t));

    cudaMalloc(&m->e,     MATRIX_MAX_D * MATRIX_MAX_V * sizeof(float_t));
    cudaMalloc(&m->spin,  MATRIX_MAX_N * MATRIX_MAX_D * MATRIX_MAX_V * sizeof(int8_t));

    return m;
}
static __inline__ void device_matrix_free(const device_submatrix_t *m) {
    cudaFree(m->ids_i);

    cudaFree(m->J);
    cudaFree(m->h);

    cudaFree(m->e);
    cudaFree(m->spin);
}

#endif //MATRIXCORE_DEVICE_MATRIX_CUH