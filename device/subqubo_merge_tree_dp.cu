// subqubo_merge_tree_dp.cu

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// ------------------------- your current effective struct -------------------------
typedef struct {
    int32_t *n;        // device pointer; n[0] = N
    int64_t *ids_i;    // device pointer; length N (global ids)
    int64_t *Q;        // unused here
    float   *J;        // unused here
    int64_t *h;        // unused here

    int8_t  *spin;     // device pointer; length N*B ; spin[b*N + i] in {-1,+1}
    float   *e;        // device pointer; length B ; energy per candidate

    int32_t  n_iter;
    float    dt;
    float    clamp;
    uint32_t seed;
} device_matrix_t;

// ------------------------- CUDA error helper -------------------------
#define CUDA_CHECK(call) do {                                        \
    cudaError_t _e = (call);                                         \
    if (_e != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                cudaGetErrorString(_e), __FILE__, __LINE__);         \
        return _e;                                                   \
    }                                                               \
} while(0)

// =============================================================================
// 1) OPTIONAL BUT RECOMMENDED: Pack energies into one contiguous device array
//    so you avoid m cudaMemcpy calls.
// =============================================================================

__global__ void pack_energies_kernel(const device_matrix_t *mats,
                                     int32_t m, int32_t B,
                                     float *outE /* m*B */)
{
    const int32_t p = blockIdx.x;
    const int32_t j = blockIdx.y * blockDim.x + threadIdx.x;
    if (p >= m || j >= B) return;
    outE[(int64_t)p * (int64_t)B + (int64_t)j] = mats[p].e[j];
}

// =============================================================================
// 2) CPU DP tree merge (energies only) with backpointers
// =============================================================================

typedef struct {
    int32_t left;   // node id or -1 for leaf
    int32_t right;  // node id or -1 for leaf
    int32_t block;  // leaf: block id; internal: -1
    int32_t K;      // number of valid entries stored (<= topN)
} dp_node_hdr_t;

typedef struct {
    dp_node_hdr_t *hdr; // [max_nodes]
    double        *E;   // [max_nodes * topN]   energies
    int32_t       *A;   // [max_nodes * topN]   leaf: sample index; internal: left entry index
    int32_t       *B;   // [max_nodes * topN]   internal: right entry index; leaf: unused
    int32_t        max_nodes;
    int32_t        topN;
    int32_t        node_count;
} dp_tree_t;

static inline void iswap_i32(int32_t *x, int32_t *y) {
    int32_t t = *x; *x = *y; *y = t;
}
static inline void dswap(double *x, double *y) {
    double t = *x; *x = *y; *y = t;
}

static int32_t select_topk_sorted(const float *e, int32_t B, int32_t K,
                                  double *outE /*len K*/, int32_t *outIdx /*len K*/)
{
    const double INF = 1.0e300;

    int32_t k = 0;
    // Fill with INF
    for (int32_t t = 0; t < K; ++t) { outE[t] = INF; outIdx[t] = -1; }

    for (int32_t j = 0; j < B; ++j) {
        double val = (double)e[j];
        int32_t id = j;

        if (k < K) {
            outE[k] = val;
            outIdx[k] = id;
            k++;
            // insertion sort step
            int32_t pos = k - 1;
            while (pos > 0 && outE[pos] < outE[pos - 1]) {
                dswap(&outE[pos], &outE[pos - 1]);
                iswap_i32(&outIdx[pos], &outIdx[pos - 1]);
                pos--;
            }
        } else {
            // K already full; only insert if better than the worst
            if (val >= outE[K - 1]) continue;
            outE[K - 1] = val;
            outIdx[K - 1] = id;
            int32_t pos = K - 1;
            while (pos > 0 && outE[pos] < outE[pos - 1]) {
                dswap(&outE[pos], &outE[pos - 1]);
                iswap_i32(&outIdx[pos], &outIdx[pos - 1]);
                pos--;
            }
        }
    }
    // number of valid entries is min(B,K)
    return (B < K) ? B : K;
}

// ----- min-heap for k-smallest pair sums of two sorted arrays -----
typedef struct {
    double  val;
    int32_t i;
    int32_t j;
} heap_item_t;

static inline void heap_swap(heap_item_t *h, int32_t a, int32_t b) {
    heap_item_t t = h[a]; h[a] = h[b]; h[b] = t;
}

static void heap_sift_up(heap_item_t *h, int32_t idx) {
    while (idx > 0) {
        const int32_t p = (idx - 1) >> 1;
        if (h[p].val <= h[idx].val) break;
        heap_swap(h, p, idx);
        idx = p;
    }
}

static void heap_sift_down(heap_item_t *h, int32_t n, int32_t idx) {
    for (;;) {
        int32_t l = (idx << 1) + 1;
        int32_t r = l + 1;
        int32_t s = idx;
        if (l < n && h[l].val < h[s].val) s = l;
        if (r < n && h[r].val < h[s].val) s = r;
        if (s == idx) break;
        heap_swap(h, s, idx);
        idx = s;
    }
}

static void heap_push(heap_item_t *h, int32_t *n, heap_item_t it) {
    int32_t idx = *n;
    h[idx] = it;
    (*n)++;
    heap_sift_up(h, idx);
}

static heap_item_t heap_pop(heap_item_t *h, int32_t *n) {
    heap_item_t out = h[0];
    (*n)--;
    if (*n > 0) {
        h[0] = h[*n];
        heap_sift_down(h, *n, 0);
    }
    return out;
}

// Merge two sorted lists EA (Ka) and EB (Kb) => topN smallest sums.
// Outputs are sorted in ascending order.
static int32_t merge_two_sorted_topn(const double *EA, int32_t Ka,
                                     const double *EB, int32_t Kb,
                                     int32_t topN,
                                     double *Eout,
                                     int32_t *ia_out,
                                     int32_t *ib_out)
{
    if (Ka <= 0 || Kb <= 0 || topN <= 0) return 0;

    // Heap of size Ka (each i starts with j=0)
    heap_item_t *heap = (heap_item_t*)malloc((size_t)Ka * sizeof(heap_item_t));
    if (!heap) return 0;

    int32_t hn = 0;
    for (int32_t i = 0; i < Ka; ++i) {
        heap_item_t it;
        it.i = i;
        it.j = 0;
        it.val = EA[i] + EB[0];
        heap_push(heap, &hn, it);
    }

    int32_t outK = 0;
    while (outK < topN && hn > 0) {
        heap_item_t it = heap_pop(heap, &hn);
        Eout[outK]   = it.val;
        ia_out[outK] = it.i;
        ib_out[outK] = it.j;
        outK++;

        int32_t nj = it.j + 1;
        if (nj < Kb) {
            heap_item_t it2;
            it2.i = it.i;
            it2.j = nj;
            it2.val = EA[it.i] + EB[nj];
            heap_push(heap, &hn, it2);
        }
    }

    free(heap);
    return outK;
}

static void dp_tree_free(dp_tree_t *T) {
    if (!T) return;
    free(T->hdr);
    free(T->E);
    free(T->A);
    free(T->B);
    memset(T, 0, sizeof(*T));
}

// Build balanced merge tree and return root node id.
static int32_t dp_tree_build(const float *allE /*m*B on host*/,
                             const int32_t m, int32_t B, const int32_t topN,
                             dp_tree_t *outT)
{
    memset(outT, 0, sizeof(*outT));

    int32_t max_nodes = 2 * m;
    outT->hdr = (dp_node_hdr_t*)malloc(max_nodes * sizeof(dp_node_hdr_t));
    outT->E   = (double*)malloc(max_nodes * topN * sizeof(double));
    outT->A   = (int32_t*)malloc(max_nodes * topN * sizeof(int32_t));
    outT->B   = (int32_t*)malloc(max_nodes * topN * sizeof(int32_t));

    if (!outT->hdr || !outT->E || !outT->A || !outT->B) {
        dp_tree_free(outT);
        return -1;
    }

    outT->max_nodes  = max_nodes;
    outT->topN       = topN;
    outT->node_count = 0;

    // Create leaf nodes [0..m-1]
    for (int32_t p = 0; p < m; ++p) {
        const int32_t node = outT->node_count++;
        outT->hdr[node].left  = -1;
        outT->hdr[node].right = -1;
        outT->hdr[node].block = p;

        double  *Eleaf = &outT->E[(int64_t)node * topN];
        int32_t *Aleaf = &outT->A[(int64_t)node * topN];
        // B leaf unused

        int32_t Kleaf = select_topk_sorted(&allE[(int64_t)p * B], B, topN, Eleaf, Aleaf);
        outT->hdr[node].K = Kleaf;
    }

    // Current frontier of node ids
    int32_t *cur = (int32_t*)malloc((size_t)m * sizeof(int32_t));
    int32_t *nxt = (int32_t*)malloc((size_t)m * sizeof(int32_t));
    if (!cur || !nxt) {
        free(cur); free(nxt);
        dp_tree_free(outT);
        return -1;
    }
    for (int32_t i = 0; i < m; ++i) cur[i] = i;
    int32_t curN = m;

    // Merge levels
    while (curN > 1) {
        int32_t nxtN = 0;
        for (int32_t t = 0; t < curN; t += 2) {
            if (t + 1 >= curN) {
                // carry odd node
                nxt[nxtN++] = cur[t];
                continue;
            }

            int32_t L = cur[t];
            int32_t R = cur[t + 1];

            int32_t node = outT->node_count++;
            outT->hdr[node].left  = L;
            outT->hdr[node].right = R;
            outT->hdr[node].block = -1;

            double  *Eout = &outT->E[(int64_t)node * topN];
            int32_t *Aout = &outT->A[(int64_t)node * topN];
            int32_t *Bout = &outT->B[(int64_t)node * topN];

            const double *EL = &outT->E[(int64_t)L * topN];
            const double *ER = &outT->E[(int64_t)R * topN];
            int32_t KL = outT->hdr[L].K;
            int32_t KR = outT->hdr[R].K;

            int32_t K = merge_two_sorted_topn(EL, KL, ER, KR, topN, Eout, Aout, Bout);
            outT->hdr[node].K = K;

            nxt[nxtN++] = node;
        }

        // swap cur/nxt
        int32_t *tmp = cur; cur = nxt; nxt = tmp;
        curN = nxtN;
    }

    int32_t root = cur[0];
    free(cur);
    free(nxt);
    return root;
}

// Decode root topK solutions => choice[block * topK + s] = sampleIndexInThatBlock
static void dp_tree_decode_choices(const dp_tree_t *T,
                                   int32_t root,
                                   int32_t m,
                                   int32_t *choice /*m*topK*/)
{
    int32_t topK = T->hdr[root].K;

    // Stack depth is about log2(m); 64 is plenty for huge m
    int32_t stack_node[64];
    int32_t stack_ent[64];

    for (int32_t s = 0; s < topK; ++s) {
        int32_t sp = 0;
        stack_node[sp] = root;
        stack_ent[sp]  = s;
        sp++;

        while (sp > 0) {
            sp--;
            int32_t node = stack_node[sp];
            int32_t ent  = stack_ent[sp];

            int32_t L = T->hdr[node].left;
            int32_t R = T->hdr[node].right;

            if (L < 0) {
                // leaf
                int32_t block = T->hdr[node].block;
                int32_t sample_idx = T->A[(int64_t)node * T->topN + ent];
                choice[(int64_t)block * topK + s] = sample_idx;
            } else {
                int32_t le = T->A[(int64_t)node * T->topN + ent];
                int32_t re = T->B[(int64_t)node * T->topN + ent];
                // push children
                stack_node[sp] = L; stack_ent[sp] = le; sp++;
                stack_node[sp] = R; stack_ent[sp] = re; sp++;
                // If you ever hit sp>=64, just enlarge arrays.
            }
        }
    }
}

// =============================================================================
// 3) GPU gather: build global spin vectors from per-block chosen candidate index
// =============================================================================

__global__ void gather_global_spins_kernel(const device_matrix_t *mats,
                                           int32_t m,
                                           int32_t topK,
                                           const int32_t *choice /*m*topK*/,
                                           int8_t *out_spin /*topK*N_total*/,
                                           int64_t N_total)
{
    int32_t p = (int32_t)blockIdx.x;
    int32_t s = (int32_t)blockIdx.y;
    if (p >= m || s >= topK) return;

    int32_t N = mats[p].n[0];
    int32_t picked = choice[(int64_t)p * topK + s];

    const int8_t  *spin = mats[p].spin;
    const int64_t *ids  = mats[p].ids_i;

    // scatter this block's spins into the global vector
    for (int32_t i = (int32_t)threadIdx.x; i < N; i += (int32_t)blockDim.x) {
        int64_t gid = ids[i]; // global index
        int8_t  v   = spin[(int64_t)picked * N + i];
        out_spin[(int64_t)s * N_total + gid] = v;
    }
}

// =============================================================================
// 4) One “do it all” helper: pack energies -> DP on CPU -> gather spins on GPU
// =============================================================================

cudaError_t merge_subqubos_topn(const device_matrix_t *d_mats,
                               int32_t m,
                               int32_t B,
                               int32_t topN,
                               int64_t N_total,
                               // outputs:
                               int8_t  *d_out_spin,   // device: topK*N_total
                               float   *h_out_energy, // host: topK (optional, can be NULL)
                               int32_t *d_out_choice, // device: m*topK (optional, can be NULL)
                               int32_t *h_out_choice, // host: m*topK (optional, can be NULL)
                               int32_t *out_topK)      // host: actual returned K (<=topN)
{
    if (m <= 0 || B <= 0 || topN <= 0) return cudaErrorInvalidValue;

    // Pack energies on device
    float *d_allE = NULL;
    CUDA_CHECK(cudaMalloc(&d_allE, m * B * sizeof(float)));

    dim3 blk(256, 1, 1);
    dim3 grd(m, (B + 255) / 256, 1);
    pack_energies_kernel<<<grd, blk>>>(d_mats, m, B, d_allE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy packed energies to host
    float *h_allE = (float*)malloc(m * B * sizeof(float));
    if (!h_allE) {
        cudaFree(d_allE);
        return cudaErrorMemoryAllocation;
    }
    CUDA_CHECK(cudaMemcpy(h_allE, d_allE, m * B * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // DP tree on CPU
    dp_tree_t T;
    int32_t root = dp_tree_build(h_allE, m, B, topN, &T);
    free(h_allE);
    cudaFree(d_allE);

    if (root < 0) return cudaErrorUnknown;

    int32_t topK = T.hdr[root].K;
    if (out_topK) *out_topK = topK;

    // Root energies to host (optional)
    if (h_out_energy) {
        const double *Er = &T.E[(int64_t)root * T.topN];
        for (int32_t s = 0; s < topK; ++s) h_out_energy[s] = (float)Er[s];
    }

    // Decode choices on host
    int32_t *h_choice = (int32_t*)malloc(m * topK * sizeof(int32_t));
    if (!h_choice) {
        dp_tree_free(&T);
        return cudaErrorMemoryAllocation;
    }
    dp_tree_decode_choices(&T, root, m, h_choice);

    // Optionally return host choices
    if (h_out_choice) {
        memcpy(h_out_choice, h_choice, m * topK * sizeof(int32_t));
    }

    // Copy choices to device (either into user buffer or temp)
    int32_t *d_choice = d_out_choice;
    if (!d_choice) {
        CUDA_CHECK(cudaMalloc(&d_choice, m * topK * sizeof(int32_t)));
    }
    CUDA_CHECK(cudaMemcpy(d_choice, h_choice,
                          (size_t)m * (size_t)topK * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    // Gather spins on GPU
    dim3 grd2((unsigned)m, (unsigned)topK, 1);
    gather_global_spins_kernel<<<grd2, 256>>>(d_mats, m, topK, d_choice, d_out_spin, N_total);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // cleanup
    free(h_choice);
    dp_tree_free(&T);

    if (!d_out_choice) cudaFree(d_choice);
    return cudaSuccess;
}

#ifdef __cplusplus
} // extern "C"
#endif