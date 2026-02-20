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
    const auto code = static_cast<bytecode_t *>(malloc(sizeof(bytecode_t)));

    code->data = nullptr;
    code->len = 0;
    code->cap = 0;

    limit_list_init(&code->limits);

    return code;
}
void bytecode_clear(bytecode_t *code) {
    if (code == nullptr) return;
    bytecode_resize(code, 0);
    limit_list_clear(&code->limits);
}
void bytecode_free(bytecode_t *code) {
    if (code == nullptr) return;

    if (code->data != nullptr) free(code->data);
    limit_list_free(&code->limits);

    free(code);
}

void bytecode_set(bytecode_t *code, const bytecode_t *src) {
    if (code == nullptr) return;
    if (src == nullptr) {
        bytecode_clear(code);
        return;
    }

    bytecode_resize(code, src->len);
    for (uint64_t i = 0; i < src->len; ++i)
        code->data[i] = src->data[i];

    limit_list_set(&code->limits, &src->limits);
}
void bytecode_addend_op(bytecode_t *code, uint8_t op) {
    if (code == nullptr) return;
    const uint64_t len = code->len;
    bytecode_resize(code, len + 1);
    code->data[len] = op;
}
void bytecode_addend_val(bytecode_t *code, int64_t val) {
    if (code == nullptr) return;
    const uint64_t len = code->len;
    bytecode_resize(code, len + sizeof(int64_t));
    *(int64_t *)(code->data + len) = val;
}
void bytecode_pop(bytecode_t *code) {
    if (code == nullptr) return;
    const uint64_t len = code->len - 1;

    code->data[len] = 0;
    code->len = len;
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



limit_t *limit_init() {
    const auto limit = static_cast<limit_t *>(malloc(sizeof(limit_t)));

    limit->symbol = 0;
    limit->min = 0;
    limit->max = 0;

    return limit;
}
void limit_clear(limit_t *limit) {
    if (limit == nullptr) return;
    limit->symbol = 0;
    limit->min = 0;
    limit->max = 0;
}
void limit_free(limit_t *limit) {
    if (limit == nullptr) return;
    free(limit);
}

void limit_set(limit_t *limit, const limit_t *src) {
    if (limit == nullptr) return;
    if (src == nullptr) {
        limit_clear(limit);
        return;
    }

    limit->symbol = src->symbol;
    limit->min = src->min;
    limit->max = src->max;
}


void limit_list_resize(limit_list_t *list, uint64_t cap) {
    if (list->data == nullptr && cap != 0) {
        list->cap = cap;
        list->data = static_cast<limit_t **>(malloc(sizeof(limit_t *) * cap));
        for (uint64_t i = 0; i < cap; i++) list->data[i] = nullptr;
    } else if (list->cap < cap) {
        list->data = static_cast<limit_t **>(realloc(list->data, sizeof(limit_t *) * cap * 2));
        for (uint64_t i = list->cap, l = cap * 2; i < l; i++) list->data[i] = nullptr;
        list->cap = cap * 2;
    }

    for (uint64_t i = cap, l = list->len; i < l; i++) {
        if (list->data[i] != nullptr) limit_free(list->data[i]);
        list->data[i] = nullptr;
    }
    list->len = cap;
}

void limit_list_init(limit_list_t *list) {
    if (list == nullptr) return;
    list->data = nullptr;
    list->len = 0;
    list->cap = 0;
}
void limit_list_clear(limit_list_t *list) {
    if (list == nullptr) return;
    limit_list_resize(list, 0);
}
void limit_list_free(limit_list_t *list) {
    if (list == nullptr || list->data == nullptr) return;
    limit_list_resize(list, 0);
    free(list->data);
}

void limit_list_set(limit_list_t *list, const limit_list_t *src) {
    if (list == nullptr || src == nullptr) return;
    limit_list_clear(list);

    for (uint64_t i = 0; i < src->len; i++)
        limit_set(limit_list_append(list), src->data[i]);
}

limit_t *limit_list_append(limit_list_t *list) {
    if (list == nullptr) return nullptr;
    const uint64_t len = list->len;
    limit_list_resize(list, len + 1);
    return list->data[len] = limit_init();
}
void limit_list_pop(limit_list_t *list) {
    if (list == nullptr) return;
    limit_list_resize(list, list->len - 1);
}