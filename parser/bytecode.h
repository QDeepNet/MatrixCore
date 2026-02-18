#ifndef MATRIXCORE_BYTECODE_H
#define MATRIXCORE_BYTECODE_H

#include <cstdint>

#define NEG 0x01
#define ADD 0x02
#define SUB 0x03
#define MUL 0x04
#define DIV 0x05
#define MOD 0x06
#define POW 0x07

#define SET     0x10
#define SET_I   0x11
#define SET_J   0x12
#define SET_QJ  0x13

typedef struct {
    uint8_t *data;
    uint64_t len;
    uint64_t cap;

    uint8_t count;

    uint8_t symbol[2];
    int64_t min[2];
    int64_t max[2];
} bytecode_t;


void bytecode_resize(bytecode_t *list, uint64_t cap);

void bytecode_init(bytecode_t *list);
void bytecode_clear(bytecode_t *list);
void bytecode_free(const bytecode_t *list);

void bytecode_set(bytecode_t *list, const bytecode_t *src);
void bytecode_addend_op(bytecode_t *list, uint8_t op);
void bytecode_addend_val(bytecode_t *list, int64_t val);
void bytecode_pop(bytecode_t *list);



typedef struct {
    uint8_t symbol;
    int64_t min;
    int64_t max;
} bytecode_limits_t;

void bytecode_limits_init(bytecode_limits_t *limits);
void bytecode_limits_clear(bytecode_limits_t *limits);
void bytecode_limits_free(bytecode_limits_t *limits);



#endif //MATRIXCORE_BYTECODE_H