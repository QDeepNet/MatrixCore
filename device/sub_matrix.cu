#include <cuda_runtime.h>

#include "device_bytecode.cuh"
#include "device_constructor.cuh"
#include "device_instruction.cuh"


__device__ __forceinline__ void __constructor_set_const   (const device_constructor_t *c, const uint32_t stack_top, const int64_t value) {
    c->stack[(stack_top << 10 | blockIdx.x) << 10 | threadIdx.x] = value;
}
__device__ __forceinline__ void __constructor_set_value_i (const device_constructor_t *c, const uint32_t stack_top, const device_submatrix_t *m) {
    c->stack[(stack_top << 10 | blockIdx.x) << 10 | threadIdx.x] = m->ids_i[threadIdx.x];
}
__device__ __forceinline__ void __constructor_set_value_j (const device_constructor_t *c, const uint32_t stack_top) {
    c->stack[(stack_top << 10 | blockIdx.x) << 10 | threadIdx.x] = c->offset_j + blockIdx.x;
}
__device__ __forceinline__ void __constructor_set_value_qj(const device_constructor_t *c, const uint32_t stack_top) {
    c->stack[(stack_top << 10 | blockIdx.x) << 10 | threadIdx.x] = c->q_j[blockIdx.x];
}

__device__ __forceinline__ void __constructor_neg(const device_constructor_t *c, const uint32_t stack_top) {
    c->stack[(stack_top << 10 | blockIdx.x) << 10 | threadIdx.x] *= -1;
}
__device__ __forceinline__ void __constructor_nml(const device_constructor_t *c, const uint32_t stack_top) {
    c->stack[(stack_top << 10 | blockIdx.x) << 10 | threadIdx.x] = 1 / c->stack[((stack_top - 1) << 10 | blockIdx.x) << 10 | threadIdx.x];
}
__device__ __forceinline__ void __constructor_add(const device_constructor_t *c, const uint32_t stack_top) {
    c->stack[((stack_top - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] += c->stack[(stack_top << 10 | blockIdx.x) << 10 | threadIdx.x];
}
__device__ __forceinline__ void __constructor_sub(const device_constructor_t *c, const uint32_t stack_top) {
    c->stack[((stack_top - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] -= c->stack[(stack_top << 10 | blockIdx.x) << 10 | threadIdx.x];
}
__device__ __forceinline__ void __constructor_mul(const device_constructor_t *c, const uint32_t stack_top) {
    c->stack[((stack_top - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] *= c->stack[(stack_top << 10 | blockIdx.x) << 10 | threadIdx.x];
}
__device__ __forceinline__ void __constructor_div(const device_constructor_t *c, const uint32_t stack_top) {
    c->stack[((stack_top - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] /= c->stack[(stack_top << 10 | blockIdx.x) << 10 | threadIdx.x];
}
__device__ __forceinline__ void __constructor_mod(const device_constructor_t *c, const uint32_t stack_top) {
    c->stack[((stack_top - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] %= c->stack[(stack_top << 10 | blockIdx.x) << 10 | threadIdx.x];
}
__device__ __forceinline__ void __constructor_pow(const device_constructor_t *c, const uint32_t stack_top) {
    int64_t res = 1;
    int64_t a = c->stack[((stack_top - 1) << 10 | blockIdx.x) << 10 | threadIdx.x];
    int64_t e = c->stack[(stack_top << 10 | blockIdx.x) << 10 | threadIdx.x];

    for (;e > 0; a *= a, e /= 2)
        if (e & 1) res *= a;

    c->stack[((stack_top - 1) << 10 | blockIdx.x) << 10 | threadIdx.x] = res;
}

// <<<1024, 1024>>>
// v - is current variance
__global__  void __constructor_interpreter(const device_constructor_t *c, const device_submatrix_t *m, const device_bytecode_t *b, const uint32_t v) {
    const int64_t real_i = m->ids_i[threadIdx.x];
    const int64_t real_j = c->offset_j + blockIdx.x;
    const int64_t relt_i = threadIdx.x;
    const int64_t relt_j = c->ids_j[blockIdx.x];
    const int64_t relt_p = relt_j != -1 ? relt_j : relt_i;

    __shared__ float_t h[1024];

    int64_t val = 0;

    __syncthreads();
    for (uint64_t i = 0; i < b->len; i++) {
        __syncthreads();
        const device_limit_t *desc = b->d_desc + i;
        const uint8_t *ptr = b->d_pool + desc->instr_off;

        if (desc->ndim == 1 && real_i != real_j) continue;
        if (desc->ndim == 2 && (real_i < desc->min_i || real_i >= desc->max_i)) continue;
        if (desc->ndim == 2 && (real_j < desc->min_j || real_j >= desc->max_j)) continue;
        
        uint32_t stack_top = 0;
        
        for (uint64_t pos = 0; pos < desc->instr_size;) {
            const uint8_t code = ptr[pos++];
            switch (code) {
                case NEG:
                    __constructor_neg(c, stack_top); break;
                case NMl:
                    __constructor_nml(c, stack_top); break;
                case ADD:
                    __constructor_add(c, stack_top); break;
                case SUB:
                    __constructor_sub(c, stack_top); break;
                case MUL:
                    __constructor_mul(c, stack_top); break;
                case DIV:
                    __constructor_div(c, stack_top); break;
                case MOD:
                    __constructor_mod(c, stack_top); break;
                case POW:
                    __constructor_pow(c, stack_top); break;

                case SET:
                    __constructor_set_const   (c, stack_top, ((uint64_t *)(ptr + pos))[0]);
                    pos += 8;                                           break;
                case SET_I:
                    __constructor_set_value_i (c, stack_top, m);        break;
                case SET_J:
                    __constructor_set_value_j (c, stack_top);           break;
                case SET_QI:
                    __constructor_set_const   (c, stack_top, 1);   break;
                case SET_QJ:
                    __constructor_set_value_qj(c, stack_top);           break;
                default: break;
            }
            if (code > NMl && code <= POW)      stack_top--;
            if (code > POW && code <= SET_QJ)   stack_top++;
        }

        val += c->stack[blockIdx.x << 10 | threadIdx.x];
    }

    if (relt_j != -1) h[threadIdx.x] = (float_t)val * (float_t)0.5;
    else atomicAdd(&m->h[relt_i], (float_t)val);
    __syncthreads();

    if (relt_j != relt_i) atomicAdd(&m->J[relt_p * m->n + relt_j], h[threadIdx.x]);

    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) h[threadIdx.x] += h[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) atomicAdd(&m->h[relt_j], h[0] * 0.5);
}

// add(r1, a1, a2);
// sub(r1, a1, a2);
// mul(r1, a1, a2);
// div(r1, a1, a2);
// pow(r1, a1, a2);

