#ifndef MATRIXCORE_DEVICE_BYTECODE_CUH
#define MATRIXCORE_DEVICE_BYTECODE_CUH

#include <cstdlib>
#include <cstring>

#include "../parser/bytecode.h"

typedef struct {
    uint32_t instr_off;   // offset in instruction pool
    uint32_t instr_size;  // bytes in this bytecode

    int64_t  min_i, max_i;
    int64_t  min_j, max_j;

    uint8_t  ndim;        // 1 = diagonal-only, 2 = 2D
} device_limit_t;
typedef struct {
    device_limit_t *d_desc;
    uint8_t *d_pool;

    uint64_t len;
} device_bytecode_t;

static device_bytecode_t *device_bytecode_init(const bytecode_list_t *list) {
    auto bytecode = (device_bytecode_t *)malloc(sizeof(device_bytecode_t));
    *bytecode = {};
    bytecode->len = list->len;

    uint64_t total_bytes = 0;
    for (uint64_t i = 0; i < list->len; ++i) total_bytes += list->data[i]->len;

    auto h_pool = (uint8_t*)malloc(total_bytes);
    auto h_desc = (device_limit_t*)malloc(list->len * sizeof(device_limit_t));

    for (uint64_t pos = 0, bi = 0; bi < list->len; ++bi) {
        const bytecode_t *bc = list->data[bi];
        memcpy(h_pool + pos, bc->data, bc->len);

        device_limit_t *bd = h_desc + bi;
        *bd = {};
        bd->instr_off  = pos;
        bd->instr_size = bc->len;

        if (bc->limits.len == 1) {
            bd->ndim  = 1;
            bd->min_i = bc->limits.data[0]->min;
            bd->max_i = bc->limits.data[0]->max;
        } else if (bc->limits.len >= 2) {
            bd->ndim = 2;

            const limit_t *L0 = bc->limits.data[0];
            const limit_t *L1 = bc->limits.data[1];

            bd->min_i = L0->min; bd->max_i = L0->max;
            bd->min_j = L1->min; bd->max_j = L1->max;
        }

        pos += bc->len;
    }


    cudaMalloc(&bytecode->d_pool, total_bytes);
    cudaMalloc(&bytecode->d_desc, list->len * sizeof(device_limit_t));

    cudaMemcpy(bytecode->d_pool, h_pool, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bytecode->d_desc, h_desc, list->len * sizeof(device_limit_t), cudaMemcpyHostToDevice);

    return bytecode;
}
static void device_bytecode_free(device_bytecode_t *bytecode) {
    cudaFree(bytecode->d_pool);
    cudaFree(bytecode->d_desc);
    free(bytecode);
}


#endif //MATRIXCORE_DEVICE_BYTECODE_CUH