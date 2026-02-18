#include <cstdint>
#include <cstdlib>

typedef struct {
    uint32_t total_spins; // total_number of spins in the matrix (M)
    uint32_t block_spins; // Number of spins inside each block (max, this mean at most) (x)
    uint32_t block_count; // Number of Block (Sub Qubo) (N)

    uint32_t candidates; // Number of candidates for Block (k)

    uint32_t *block_ids; // (blocks)

} data_t;


__inline__ void data_init(data_t *data) {
    data->total_spins = 0;
    data->block_spins = 0;
    data->block_count = 0;
}
__inline__ void data_clear(data_t *data) {
    data->total_spins = 0;
    data->block_spins = 0;
    data->block_count = 0;

    if (data->block_ids) free(data->block_ids);
    data->block_ids = nullptr;

}
__inline__ void data_free(data_t *data) {

}


__inline__ data_set(data_t *data, uint64_t total_spins, uint64_t block_spins) {
    data_clear(data);
    data->total_spins = total_spins;
    data->block_spins = block_spins;
    data->block_count = (total_spins + block_spins - 1) / block_spins;

    data->block_ids = static_cast<uint32_t *>(malloc(sizeof(uint32_t) * data->block_count * data->block_spins));
}