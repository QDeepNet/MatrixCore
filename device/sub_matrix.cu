#include <cuda_runtime.h>
#include "sub_matrix.cuh"

#include "device_bytecode.cuh"
#include "../parser/bytecode.h"


__device__ __forceinline__ void __constructor_set_const(const device_constructor_t *c, const int64_t value) {
    c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x] = value;
}
__device__ __forceinline__ void __constructor_set_value_i(const device_constructor_t *c, const device_matrix_t *m) {
    c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x] = m->ids_i[threadIdx.x];
}
__device__ __forceinline__ void __constructor_set_value_j(const device_constructor_t *c) {
    c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x] = c->offset_j + blockIdx.x;
}
__device__ __forceinline__ void __constructor_set_value_qj(const device_constructor_t *c) {
    c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x] = c->q_j[blockIdx.x];
}
__device__ __forceinline__ void __constructor_put_QIJ(const device_constructor_t *c, const device_matrix_t *m, instruction_params_t *p) {
    int64_t val = 0;
    int64_t offset_j = c->offset_j;
    __shared__ int64_t values[1024];

    int64_t i  = m->ids_i[threadIdx.x];
    int64_t j  = offset_j + blockIdx.x;
    int64_t ij = c->ids_j[blockIdx.x];

    if (p->min_i > i || i > p->max_i) goto sum;
    if (p->min_j > j || j > p->max_j) goto sum;
    if (ij == -1) goto sum;

    atomicAdd(reinterpret_cast<unsigned long long *>(m->Q) + m->n * ij + threadIdx.x, c->stack[blockIdx.x << 10 | threadIdx.x]);

    sum:
    __syncthreads();


    i = m->ids_i[blockIdx.x];
    j = offset_j + threadIdx.x;
    ij = c->ids_j[threadIdx.x];


    if (p->min_i > i || i > p->max_i) goto next;
    if (p->min_j > j || j > p->max_j) goto next;
    if (ij != -1) goto next;

    val = c->stack[threadIdx.x << 10 | blockIdx.x];

    next:
    values[threadIdx.x] = val;
    __syncthreads();

    for (unsigned int stride = 1024 / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) values[threadIdx.x] += values[threadIdx.x + stride];
        __syncthreads();
    }

    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(reinterpret_cast<unsigned long long *>(m->Q) + m->n * threadIdx.x + threadIdx.x, values[0]);
}


__device__ __forceinline__ void __constructor_neg(const device_constructor_t *c) {
    c->stack[((c->stack_top[blockIdx.x] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] *= -1;
}
__device__ __forceinline__ void __constructor_add(const device_constructor_t *c) {
    c->stack[((c->stack_top[blockIdx.x] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] += c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x];
}
__device__ __forceinline__ void __constructor_sub(const device_constructor_t *c) {
    c->stack[((c->stack_top[blockIdx.x] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] -= c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x];
}
__device__ __forceinline__ void __constructor_nml(const device_constructor_t *c) {
    c->stack[((c->stack_top[blockIdx.x] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] = 1 / c->stack[((c->stack_top[blockIdx.x] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x];
}
__device__ __forceinline__ void __constructor_mul(const device_constructor_t *c) {
    c->stack[((c->stack_top[blockIdx.x] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] *= c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x];
}
__device__ __forceinline__ void __constructor_div(const device_constructor_t *c) {
    c->stack[((c->stack_top[blockIdx.x] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] /= c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x];
}
__device__ __forceinline__ void __constructor_mod(const device_constructor_t *c) {
    c->stack[((c->stack_top[blockIdx.x] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] %= c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x];
}
__device__ __forceinline__ void __constructor_pow(const device_constructor_t *c) {
    int64_t res = 1;
    int64_t a = c->stack[((c->stack_top[blockIdx.x] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x];
    int64_t e = c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x];

    for (;e > 0; a *= a, e /= 2)
        if (e & 1) res *= a;

    c->stack[((c->stack_top[blockIdx.x] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] = res;
}


__global__  void __constructor_interpreter(const device_constructor_t *c, const device_matrix_t *m, const device_bytecode_t *b) {
    const int64_t real_i = m->ids_i[threadIdx.x];
    const int64_t real_j = c->offset_j + blockIdx.x;
    const int64_t relt_j = c->ids_j[blockIdx.x];

    __shared__ int64_t val[1024];

    val[threadIdx.x] = 0;

    __syncthreads();
    for (uint64_t i = 0; i < b->len; i++) {
        const device_limit_t *desc = b->d_desc + i;
        const uint8_t *ptr = b->d_pool + desc->instr_off;

        uint8_t good = 1;
        if (desc->ndim == 1 && real_i != real_j) good = 0;
        if (desc->ndim == 2 && (real_i < desc->min_i || real_i >= desc->max_i)) good = 0;
        if (desc->ndim == 2 && (real_j < desc->min_j || real_j >= desc->max_j)) good = 0;


        for (uint64_t pos = 0; pos < desc->instr_size;) {
            const uint8_t code = ptr[pos++];
            if (good) switch (code) {
                case NEG:
                    __constructor_neg(c); break;
                case NMl:
                    __constructor_nml(c); break;
                case ADD:
                    __constructor_add(c); break;
                case SUB:
                    __constructor_sub(c); break;
                case MUL:
                    __constructor_mul(c); break;
                case DIV:
                    __constructor_div(c); break;
                case MOD:
                    __constructor_mod(c); break;
                case POW:
                    __constructor_pow(c); break;

                case SET:
                    __constructor_set_const(c, ((uint64_t *)(ptr + pos))[0]);
                    pos += 8;                           break;
                case SET_I:
                    __constructor_set_value_i(c, m);    break;
                case SET_J:
                    __constructor_set_value_j(c);       break;
                case SET_QI:
                    __constructor_set_const(c, 1); break;
                case SET_QJ:
                    __constructor_set_value_qj(c);      break;
                default: break;
            }
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                if (code > NMl && code <= POW)      atomicSub(c->stack_top + blockIdx.x, 1);
                if (code > POW && code <= SET_QJ)   atomicAdd(c->stack_top + blockIdx.x, 1);
            }
        }

        __syncthreads();
        if (good) val[threadIdx.x] += c->stack[threadIdx.x << 10 | blockIdx.x];
        if (threadIdx.x == 0 && blockIdx.x == 0) atomicSub(c->stack_top + blockIdx.x, 1);
    }

    __syncthreads();
    atomicAdd((uint64_t *)m->Q + m->n * (relt_j != -1 ? relt_j : threadIdx.x) + threadIdx.x, val[threadIdx.x]);
}

// add(r1, a1, a2);
// sub(r1, a1, a2);
// mul(r1, a1, a2);
// div(r1, a1, a2);
// pow(r1, a1, a2);

