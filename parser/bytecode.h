#ifndef MATRIXCORE_BYTECODE_H
#define MATRIXCORE_BYTECODE_H

#include <cstdint>

#define NEG 0x01
#define ADD 0x02
#define SUB 0x03
#define NMl 0x04
#define MUL 0x05
#define DIV 0x06
#define MOD 0x07
#define POW 0x08

#define SET     0x10
#define SET_I   0x11
#define SET_J   0x12
#define SET_QI  0x13
#define SET_QJ  0x14



typedef struct {
    uint8_t symbol;
    int64_t min;
    int64_t max;
} limit_t;

typedef struct {
    limit_t **data;
    uint64_t len;
    uint64_t cap;
} limit_list_t;

typedef struct {
    uint8_t *data;
    uint64_t len;
    uint64_t cap;

    limit_list_t limits;
} bytecode_t;

typedef struct {
    bytecode_t **data;
    uint64_t len;
    uint64_t cap;
} bytecode_list_t;


void bytecode_resize(bytecode_t *code, uint64_t cap);

bytecode_t *bytecode_init();
void bytecode_clear(bytecode_t *code);
void bytecode_free(bytecode_t *code);

void bytecode_set(bytecode_t *code, const bytecode_t *src);
void bytecode_addend_op(bytecode_t *code, uint8_t op);
void bytecode_addend_val(bytecode_t *code, int64_t val);
void bytecode_pop(bytecode_t *code);


void bytecode_list_init(bytecode_list_t *list);
void bytecode_list_clear(bytecode_list_t *list);
void bytecode_list_free(bytecode_list_t *list);

void bytecode_list_move(bytecode_list_t *list, bytecode_list_t *src);

bytecode_t *bytecode_list_append(bytecode_list_t *list);
void bytecode_list_delete(bytecode_list_t *list, uint64_t id);
void bytecode_list_pop(bytecode_list_t *list);


limit_t *limit_init();
void limit_clear(limit_t *limit);
void limit_free(limit_t *limit);

void limit_set(limit_t *limit, const limit_t *src);


void limit_list_init(limit_list_t *list);
void limit_list_clear(limit_list_t *list);
void limit_list_free(limit_list_t *list);

void limit_list_set(limit_list_t *list, const limit_list_t *src);

limit_t *limit_list_append(limit_list_t *list);
void limit_list_pop(limit_list_t *list);



#endif //MATRIXCORE_BYTECODE_H