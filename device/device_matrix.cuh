#ifndef MATRIXCORE_DEVICE_MATRIX_CUH
#define MATRIXCORE_DEVICE_MATRIX_CUH

#include <stdint.h>
#include <stdlib.h>


typedef struct {
    int32_t n;
    int32_t d;      // Number of solution from the Solver
    int64_t *ids_i; // indexes to make sub qubo
    int64_t *Q;     // n x n (row-major) Qubo

    float_t *J;     // n x n (row-major) Ising
    int64_t *h;     // optional

    // ---- CFC state/output on device ----
    int8_t  *spin;    // optional, size = n, writes +/-1 if non-null

    // ---- solver params ----
    int32_t  n_iter;  // iterations
    float_t  dt;      // typically 0.1f
    float_t  clamp;   // typically 1.5f
    uint32_t seed;    // for optional init
} device_matrix_t;


static __inline__ device_matrix_t *device_matrix_init(const uint64_t n, const uint64_t d) {
    const auto m = static_cast<device_matrix_t *>(malloc(sizeof(device_matrix_t)));
    cudaMalloc(&m->ids_i, n * sizeof(int64_t));

    cudaMalloc(&m->Q, n * n * sizeof(int64_t));
    cudaMalloc(&m->J, n * n * sizeof(int64_t));
    cudaMalloc(&m->h, n * sizeof(int64_t));

    cudaMalloc(&m->spin, n * d * sizeof(int8_t));

    return m;
}
static __inline__ void device_matrix_free(device_matrix_t *m) {
    cudaFree(m->ids_i);

    cudaFree(m->Q);
    cudaFree(m->J);
    cudaFree(m->h);
}

#endif //MATRIXCORE_DEVICE_MATRIX_CUH