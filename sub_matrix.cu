#include <cuda_runtime.h>
#include "sub_matrix.cuh"


#define ADD 0x01
#define SUB 0x02
#define MUL 0x03
#define DIV 0x04
#define MOD 0x05
#define POW 0x06

#define SET     0x10
#define SET_I   0x11
#define SET_J   0x12
#define SET_QJ  0x13

#define PUT     0x20

__device__ void __constructor_set_const(device_matrix_t *m, const int64_t value) {
    m->stack[(m->top[0] << 10 | blockIdx.x) << 10 | threadIdx.x] = value;
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(m->top, 1);
}
__device__ void __constructor_set_value_i(device_matrix_t *m) {
    m->stack[(m->top[0] << 10 | blockIdx.x) << 10 | threadIdx.x] = m->ids_i[threadIdx.x];
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(m->top, 1);
}
__device__ void __constructor_set_value_j(device_matrix_t *m) {
    m->stack[(m->top[0] << 10 | blockIdx.x) << 10 | threadIdx.x] = m->data->offset_j + blockIdx.x;
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(m->top, 1);
}
__device__ void __constructor_set_value_qj(device_matrix_t *m) {
    m->stack[(m->top[0] << 10 | blockIdx.x) << 10 | threadIdx.x] = m->q_j[blockIdx.x];
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicAdd(m->top, 1);
}
__device__ void __constructor_put(device_matrix_t *m, instruction_params_t *p) {
    int64_t val = 0;
    int64_t offset_i = m->data->offset_i;
    int64_t offset_j = m->data->offset_j;
    __shared__ int64_t values[1024];

    uint64_t i  = offset_i + m->ids_i[threadIdx.x];
    uint64_t j  = offset_j + blockIdx.x;
    uint64_t ij = m->ids_j[blockIdx.x];

    if (p->min_i > i || i > p->max_i) goto sum;
    if (p->min_j > j || j > p->max_j) goto sum;
    if (ij == -1) goto sum;

    atomicAdd((unsigned long long *)m->matrix + m->data->n * ij + threadIdx.x, m->stack[blockIdx.x << 10 | threadIdx.x]);

    sum:
    __syncthreads();


    i = offset_i + m->ids_i[blockIdx.x];
    j = offset_j + threadIdx.x;
    ij = m->ids_j[threadIdx.x];


    if (p->min_i > i || i > p->max_i) goto sum;
    if (p->min_j > j || j > p->max_j) goto sum;
    if (ij != -1) goto sum;

    val = m->stack[threadIdx.x << 10 | blockIdx.x];

    next:
    values[threadIdx.x] = val;
    __syncthreads();

    for (unsigned int stride = threadIdx.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) values[threadIdx.x] += values[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd((unsigned long long *)m->matrix + m->data->n * threadIdx.x + threadIdx.x, values[0]);
}


__device__ void __constructor_add(device_matrix_t *m) {
    m->stack[((m->top[0] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] += m->stack[(m->top[0] << 10 | blockIdx.x) << 10 | threadIdx.x];
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicSub(m->top, 1);
}
__device__ void __constructor_sub(device_matrix_t *m) {
    m->stack[((m->top[0] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] -= m->stack[(m->top[0] << 10 | blockIdx.x) << 10 | threadIdx.x];
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicSub(m->top, 1);
}
__device__ void __constructor_mul(device_matrix_t *m) {
    m->stack[((m->top[0] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] *= m->stack[(m->top[0] << 10 | blockIdx.x) << 10 | threadIdx.x];
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicSub(m->top, 1);
}
__device__ void __constructor_div(device_matrix_t *m) {
    m->stack[((m->top[0] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] /= m->stack[(m->top[0] << 10 | blockIdx.x) << 10 | threadIdx.x];
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicSub(m->top, 1);
}
__device__ void __constructor_mod(device_matrix_t *m) {
    m->stack[((m->top[0] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] %= m->stack[(m->top[0] << 10 | blockIdx.x) << 10 | threadIdx.x];
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicSub(m->top, 1);
}
__device__ void __constructor_pow(device_matrix_t *m) {
    int64_t res = 1;
    int64_t a = m->stack[((m->top[0] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x];
    int64_t e = m->stack[(m->top[0] << 10 | blockIdx.x) << 10 | threadIdx.x];

    while (e > 0) {
        if (e & 1) res *= a;
        a *= a;
        e /= 2;
    }

    m->stack[((m->top[0] - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] = res;
    if (threadIdx.x == 0 && blockIdx.x == 0) atomicSub(m->top, 1);
}


__device__ void __constructor_interpreter(device_matrix_t *m, instruction_params_t *p, device_instruction *list, const uint64_t size) {
    for (uint64_t i = 0; i < size;) {

        switch (list[i++]) {
            case ADD:
                __constructor_add(m); break;
            case SUB:
                __constructor_sub(m); break;
            case MUL:
                __constructor_mul(m); break;
            case DIV:
                __constructor_div(m); break;
            case MOD:
                __constructor_mod(m); break;
            case POW:
                __constructor_pow(m); break;

            case SET:
                __constructor_set_const(m, ((uint64_t *)(list + i))[0]);
                i += 8;
                break;
            case SET_I:
                __constructor_set_value_i(m);break;
            case SET_J:
                __constructor_set_value_j(m);break;
            case SET_QJ:
                __constructor_set_value_qj(m);break;

            case PUT:
                __constructor_put(m, p); break;
            default: break;
        }
    }
}

// add(r1, a1, a2);
// sub(r1, a1, a2);
// mul(r1, a1, a2);
// div(r1, a1, a2);
// pow(r1, a1, a2);

