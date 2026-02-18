#include "bytecode.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>


void bytecode_resize(bytecode_t *list, uint64_t cap) {
    if (list->data == nullptr && cap != 0) {
        list->cap = cap;
        list->data = static_cast<uint8_t *>(malloc(sizeof(uint8_t) * cap));
        for (uint64_t i = 0; i < cap; i++) list->data[i] = 0;
    } else if (list->cap < cap) {
        list->data = static_cast<uint8_t *>(realloc(list->data, sizeof(uint8_t) * cap * 2));
        for (uint64_t i = list->cap, l = cap * 2; i < l; i++) list->data[i] = 0;
        list->cap = cap * 2;
    }
    list->len = cap;
}

void bytecode_init(bytecode_t *list) {
    if (list == nullptr) return;
    list->count = 0;
    memset(list->symbol, 0, sizeof(uint8_t) * 2);
    memset(list->min, 0, sizeof(int64_t) * 2);
    memset(list->max, 0, sizeof(int64_t) * 2);

    list->data = nullptr;
    list->len = 0;
    list->cap = 0;
}
void bytecode_clear(bytecode_t *list) {
    if (list == nullptr) return;
    bytecode_resize(list, 0);
    list->count = 0;
    memset(list->symbol, 0, sizeof(uint8_t) * 2);
    memset(list->min, 0, sizeof(int64_t) * 2);
    memset(list->max, 0, sizeof(int64_t) * 2);
}
void bytecode_free(const bytecode_t *list) {
    if (list == nullptr || list->data == nullptr) return;
    free(list->data);
}

void bytecode_set(bytecode_t *list, const bytecode_t *src) {
    if (list == nullptr) return;
    if (src == nullptr) {
        bytecode_clear(list);
        return;
    }

    memcpy(list->symbol, src->symbol, sizeof(uint8_t) * 2);
    memcpy(list->min, src->min, sizeof(int64_t) * 2);
    memcpy(list->max, src->max, sizeof(int64_t) * 2);

    bytecode_resize(list, src->len);
    for (uint64_t i = 0; i < src->len; ++i)
        list->data[i] = src->data[i];
}
void bytecode_addend_op(bytecode_t *list, uint8_t op) {
    if (list == nullptr) return;
    const uint64_t len = list->len;
    bytecode_resize(list, len + 1);
    list->data[len] = op;
}
void bytecode_addend_val(bytecode_t *list, int64_t val) {
    if (list == nullptr) return;
    const uint64_t len = list->len;
    bytecode_resize(list, len + sizeof(int64_t));
    *(int64_t *)(list->data + len) = val;
}
void bytecode_pop(bytecode_t *list) {
    if (list == nullptr) return;
    const uint64_t len = list->len - 1;

    list->data[len] = 0;
    list->len = len;
}