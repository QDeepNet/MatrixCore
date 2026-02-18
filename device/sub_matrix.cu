#include <cuda_runtime.h>
#include "sub_matrix.cuh"
#include "../parser/bytecode.h"


__device__ void __constructor_set_const(const device_constructor_t *c, const int64_t value) {
    c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x] = value;
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(c->stack_top + blockIdx.x, 1);
}
__device__ void __constructor_set_value_i(const device_constructor_t *c, const device_matrix_t *m) {
    c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x] = m->ids_i[threadIdx.x];
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(c->stack_top + blockIdx.x, 1);
}
__device__ void __constructor_set_value_j(const device_constructor_t *c) {
    c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x] = c->offset_j[0] + blockIdx.x;
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(c->stack_top + blockIdx.x, 1);
}
__device__ void __constructor_set_value_qj(const device_constructor_t *c) {
    c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x] = c->q_j[blockIdx.x];
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(c->stack_top + blockIdx.x, 1);
}
__device__ void __constructor_put(const device_constructor_t *c, const device_matrix_t *m, instruction_params_t *p) {
    int64_t val = 0;
    int64_t offset_j = c->offset_j[0];
    __shared__ int64_t values[1024];

    int64_t i  = m->ids_i[threadIdx.x];
    int64_t j  = offset_j + blockIdx.x;
    int64_t ij = c->ids_j[blockIdx.x];

    if (p->min_i > i || i > p->max_i) goto sum;
    if (p->min_j > j || j > p->max_j) goto sum;
    if (ij == -1) goto sum;

    atomicAdd(reinterpret_cast<unsigned long long *>(m->Q) + m->n[0] * ij + threadIdx.x, c->stack[blockIdx.x << 10 | threadIdx.x]);

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

    for (unsigned int stride = threadIdx.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) values[threadIdx.x] += values[threadIdx.x + stride];
        __syncthreads();
    }

    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(reinterpret_cast<unsigned long long *>(m->Q) + m->n[0] * threadIdx.x + threadIdx.x, values[0]);
}


__device__ void __constructor_neg(const device_constructor_t *c) {
    c->stack[((c->stack_top[blockIdx.x] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] *= -1;
}
__device__ void __constructor_add(const device_constructor_t *c) {
    c->stack[((c->stack_top[blockIdx.x] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] += c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x];
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicSub(c->stack_top + blockIdx.x, 1);
}
__device__ void __constructor_sub(const device_constructor_t *c) {
    c->stack[((c->stack_top[blockIdx.x] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] -= c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x];
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicSub(c->stack_top + blockIdx.x, 1);
}
__device__ void __constructor_mul(const device_constructor_t *c) {
    c->stack[((c->stack_top[blockIdx.x] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] *= c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x];
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicSub(c->stack_top + blockIdx.x, 1);
}
__device__ void __constructor_div(const device_constructor_t *c) {
    c->stack[((c->stack_top[blockIdx.x] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] /= c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x];
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicSub(c->stack_top + blockIdx.x, 1);
}
__device__ void __constructor_mod(const device_constructor_t *c) {
    c->stack[((c->stack_top[blockIdx.x] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] %= c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x];
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicSub(c->stack_top + blockIdx.x, 1);
}
__device__ void __constructor_pow(const device_constructor_t *c) {
    int64_t res = 1;
    int64_t a = c->stack[((c->stack_top[blockIdx.x] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x];
    int64_t e = c->stack[(c->stack_top[blockIdx.x] << 10 | blockIdx.x) << 10 | threadIdx.x];

    while (e > 0) {
        if (e & 1) res *= a;
        a *= a;
        e /= 2;
    }

    c->stack[((c->stack_top[blockIdx.x] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] = res;
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicSub(c->stack_top + blockIdx.x, 1);
}


__global__  void __constructor_interpreter(const device_constructor_t *c, const device_matrix_t *m, const device_instruction_t *i) {
    for (uint64_t _i = 0; _i < i->params->size;) {
        switch (i->instructions[_i++]) {
            case NEG:
                __constructor_neg(c); break;
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
                __constructor_set_const(c, ((uint64_t *)(i->instructions + _i))[0]);
                _i += 8;
                break;
            case SET_I:
                __constructor_set_value_i(c, m);break;
            case SET_J:
                __constructor_set_value_j(c);break;
            case SET_QJ:
                __constructor_set_value_qj(c);break;

            default: break;
        }
    }


    __constructor_put(c, m, i->params);
}

// add(r1, a1, a2);
// sub(r1, a1, a2);
// mul(r1, a1, a2);
// div(r1, a1, a2);
// pow(r1, a1, a2);

