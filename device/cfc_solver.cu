
#include "device_matrix.cuh"

__device__ __forceinline__ float clamp_abs(float v, float lim) {
    if (v >  lim) return  lim;
    if (v < -lim) return -lim;
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    unsigned mask = 0xFFFFFFFFu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v,  8);
    v += __shfl_down_sync(mask, v,  4);
    v += __shfl_down_sync(mask, v,  2);
    v += __shfl_down_sync(mask, v,  1);
    return v;
}

__device__ __forceinline__ uint32_t xorshift32(uint32_t s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s <<  5;
    return s;
}
__device__ __forceinline__ float u01_open01(uint32_t s) {
    uint32_t r = xorshift32(s);
    return ( (r >> 8) + 1.0f ) * (1.0f / 16777217.0f);
}
__device__ __forceinline__ float randn01(uint32_t s) {
    float u1 = u01_open01(s);
    float u2 = u01_open01(s);
    float r  = sqrtf(-2.0f * __logf(u1));
    float th = 6.283185307179586f * u2;
    return r * __cosf(th);
}


void cfc_solver(device_matrix_t *m) {
    const uint32_t tid  = threadIdx.x;
    const int32_t  n    = m->n[0];

    // Shared: x + tmp buffer (tmp is reused: reduction + x_next)
    __shared__ float_t x_sh[1024];
    __shared__ float_t e_sh[1024];
    __shared__ float_t tmp[1024];
    __shared__ float_t xi = 0;

    // -------------------------
    // 1) compute xi = sqrt(2n / sum(J^2))
    // -------------------------
    float_t sum = 0.0f;
    for(uint32_t i = tid; i < n * n; i += 1024) {
        const float_t v = m->J[i];
        sum += v * v;
    }

    tmp[tid] = sum;
    __syncthreads();

    for(uint32_t stride = 1024 / 2; stride > 0; stride >>= 1) {
        if(tid < stride) tmp[tid] += tmp[tid + stride];
        __syncthreads();
    }

    if(tid == 0) {
        const float_t denom = tmp[0];
        xi = sqrtf(2.0f * n / (denom < 1e-8f? 1e-8f : denom));
    }
    __syncthreads();

    // -------------------------
    // 2) load/init x into shared
    //    B (batch) is gridDim.x, sample index is blockIdx.x
    // -------------------------
    if (tid < n) {
        x_sh[tid] = 0.1f * randn01(m->seed ^ 0x9E3779B9u ^ 0x85EBCA6Bu * (tid + 1));
        e_sh[tid] = 1.0f;
    }
    __syncthreads();

    // Params (fallback to defaults if unset)
    const uint32_t lane      = tid & 31u;

    const uint32_t n_iter = m->n_iter > 0   ? m->n_iter : 1;
    const float_t  dt     = m->dt    > 0.0f ? m->dt     : 0.1f;
    const float_t  lim    = m->clamp > 0.0f ? m->clamp  : 1.5f;

    const uint32_t Tr = 10 * n_iter / 9;


    // -------------------------
    // 3) main loop
    // -------------------------
    for (uint32_t it = 0; it < n_iter; ++it) {

        // compute p[i] - 1.0f on the fly:
        // p = [linspace(-1,1,Tr), ones(Tp)]
        const float_t p_minus_1 = (Tr <= 0 || it >= Tr) ? 0 : ((Tr == 1) ? -2 : (-2 + 2 * (float_t) it / (Tr - 1)));

        for (uint32_t row = tid >> 5; row < n; row += 32) {
            const float_t *rowJ = m->J + row * n;

            // acc = dot(J[row,:], x)
            float_t acc = 0.0f;
            for (int k = (int)lane; k < n; k += 32)
                acc += rowJ[k] * x_sh[k];
            acc = warp_sum(acc);

            if (lane != 0) continue;

            const float_t x = x_sh[row];
            const float_t e = e_sh[row];
            const float_t z = xi * e * (acc + m->h[row]);

            // x = x + (-x^3 + (p-1)*x + z)*dt
            float_t x_next = x + (-(x * x * x) + p_minus_1 * x + z) * dt;

            // e = e + (-beta*e*(z^2 - alpha))*dt
            float_t e_next = e + -0.15f * e * (z * z - 1.0f) * dt;

            // clamp x and e (python where())
            x_next = clamp_abs(x_next, lim);
            if (e_next < 0.01f) e_next = 0.01f;

            tmp[row]   = x_next;
            e_sh[row]  = e_next;
        }

        __syncthreads();

        // commit x_next
        if (tid < n) x_sh[tid] = tmp[tid];

        __syncthreads();
    }

    // -------------------------
    // 4) export results to global
    // -------------------------
    if (tid < n) m->spin[tid] = x_sh[tid] >= 0.0f ? 1 : -1;
}
