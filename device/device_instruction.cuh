#ifndef MATRIXCORE_DEVICE_INSTRUCTION_CUH
#define MATRIXCORE_DEVICE_INSTRUCTION_CUH


typedef uint8_t device_instruction;
typedef struct {
    int32_t min_i;
    int32_t max_i;

    int32_t min_j;
    int32_t max_j;

    uint64_t size;
} instruction_params_t;
typedef struct {
    instruction_params_t *params;

    uint8_t *instructions;
} device_instruction_t;


static __inline__ device_instruction_t *device_instruction_init(const uint64_t size) {
    const auto i = static_cast<device_instruction_t *>(malloc(sizeof(device_instruction_t)));
    cudaMalloc(&i->params, sizeof(instruction_params_t));
    cudaMalloc(&i->instructions, size * sizeof(uint8_t));
    return i;
}
static __inline__ void device_instruction_free(device_instruction_t *i) {
    cudaFree(i->params);
    cudaFree(i->instructions);
}

#endif //MATRIXCORE_DEVICE_INSTRUCTION_CUH