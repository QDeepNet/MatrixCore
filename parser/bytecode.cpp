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

bytecode_t *bytecode_init() {
    const auto list = static_cast<bytecode_t *>(malloc(sizeof(bytecode_t)));

    list->data = nullptr;
    list->len = 0;
    list->cap = 0;

    return list;
}
void bytecode_clear(bytecode_t *list) {
    if (list == nullptr) return;
    bytecode_resize(list, 0);
}
void bytecode_free(bytecode_t *list) {
    if (list == nullptr) return;

    if (list->data != nullptr) free(list->data);
    free(list);
}

void bytecode_set(bytecode_t *list, const bytecode_t *src) {
    if (list == nullptr) return;
    if (src == nullptr) {
        bytecode_clear(list);
        return;
    }

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


void bytecode_list_resize(bytecode_list_t *list, uint64_t cap) {
    if (list->data == nullptr && cap != 0) {
        list->cap = cap;
        list->data = static_cast<bytecode_t **>(malloc(sizeof(bytecode_t *) * cap));
        for (uint64_t i = 0; i < cap; i++) list->data[i] = nullptr;
    } else if (list->cap < cap) {
        list->data = static_cast<bytecode_t **>(realloc(list->data, sizeof(bytecode_t *) * cap * 2));
        for (uint64_t i = list->cap, l = cap * 2; i < l; i++) list->data[i] = nullptr;
        list->cap = cap * 2;
    }

    for (uint64_t i = cap, l = list->len; i < l; i++) {
        if (list->data[i] != nullptr) bytecode_free(list->data[i]);
        list->data[i] = nullptr;
    }
    list->len = cap;
}

void bytecode_list_init(bytecode_list_t *list) {
    if (list == nullptr) return;
    list->data = nullptr;
    list->len = 0;
    list->cap = 0;
}
void bytecode_list_clear(bytecode_list_t *list) {
    if (list == nullptr) return;
    bytecode_list_resize(list, 0);
}
void bytecode_list_free(bytecode_list_t *list) {
    if (list == nullptr || list->data == nullptr) return;
    bytecode_list_resize(list, 0);
    free(list->data);
}

void bytecode_list_move(bytecode_list_t *list, bytecode_list_t *src) {
    if (list == nullptr) return;
    bytecode_list_clear(list);
    if (src == nullptr) return;

    bytecode_list_resize(list, src->len);
    for (uint64_t i = 0; i < src->len; ++i) {
        list->data[i] = src->data[i];
        src->data[i] = nullptr;
    }
    src->len = 0;
}

bytecode_t *bytecode_list_append(bytecode_list_t *list) {
    if (list == nullptr) return nullptr;
    const uint64_t len = list->len;
    bytecode_list_resize(list, len + 1);
    return list->data[len] = bytecode_init();
}
void bytecode_list_delete(bytecode_list_t *list, uint64_t id) {
    if (list == nullptr || id >= list->len) return;

    const uint64_t len = list->len - 1;
    bytecode_t *temp = list->data[id];
    list->data[id] = list->data[len];
    list->data[len] = temp;
    bytecode_list_resize(list, len);
}
void bytecode_list_pop(bytecode_list_t *list) {
    if (list == nullptr) return;
    bytecode_list_resize(list, list->len - 1);
}