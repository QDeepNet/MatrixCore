#include "token.h"
#include <cstdlib>

token_t *token_init() {
    const auto token = static_cast<token_t *>(malloc(sizeof(token_t)));
    token->type = TokenType_None;
    token->sub_type = TokenType_None;

    parser_data_init(&token->data);
    parser_line_init(&token->line);
    return token;
}
void token_clear(token_t *token) {
    if (token == nullptr) return;
    token->type = TokenType_None;
    token->sub_type = TokenType_None;

    parser_data_init(&token->data);
    parser_line_init(&token->line);
}
void token_free(token_t *token) {
    if (token == nullptr) return;
    free(token);
}

void token_set_data(token_t *token, const uint8_t *str_data, const uint64_t str_size) {
    if (token == nullptr) return;
    token->data.data = str_data;
    token->data.size = str_size;
}
void token_set_line(token_t *token, const parser_line_t &line) {
    if (token == nullptr) return;
    token->line = line;
}

void __token_list_resize(token_list_t *list, const uint64_t cap) {
    if (list->tokens == nullptr && cap != 0) {
        list->cap = cap;
        list->tokens = static_cast<token_t **>(malloc(sizeof(token_t *) * cap));
        for (uint64_t i = 0; i < cap; i++) list->tokens[i] = nullptr;
    } else if (list->cap < cap) {
        list->tokens = static_cast<token_t **>(realloc(list->tokens, sizeof(token_t *) * cap * 2));
        for (uint64_t i = list->cap, l = cap * 2; i < l; i++) list->tokens[i] = nullptr;
        list->cap = cap * 2;
    }
}

void token_list_resize(token_list_t *list, const uint64_t cap) {
    __token_list_resize(list, cap);

    for (uint64_t i = cap, l = list->len; i < l; i++) {
        if (list->tokens[i] != nullptr) token_free(list->tokens[i]);
        list->tokens[i] = nullptr;
    }
    list->len = cap;
}

void token_list_init(token_list_t *list) {
    if (list == nullptr) return;
    list->tokens = nullptr;
    list->len = 0;
    list->cap = 0;
}
void token_list_clear(token_list_t *list) {
    if (list == nullptr) return;
    token_list_resize(list, 0);
}
void token_list_free(token_list_t *list) {
    if (list == nullptr || list->tokens == nullptr) return;
    token_list_resize(list, 0);
    free(list->tokens);
}

token_t *token_list_append(token_list_t *list) {
    if (list == nullptr) return nullptr;
    const uint64_t len = list->len;
    token_list_resize(list, len + 1);
    return list->tokens[len] = token_init();
}
void token_list_pop(token_list_t *list) {
    if (list == nullptr) return;
    const uint64_t len = list->len - 1;

    if (list->tokens[len] != nullptr) token_free(list->tokens[len]);
    list->tokens[len] = nullptr;
    list->len = len;
}


void token_plist_resize(token_list_t *list, const uint64_t cap) {
    __token_list_resize(list, cap);
    list->len = cap;
}

void token_plist_init(token_list_t *list) {
    if (list == nullptr) return;
    list->tokens = nullptr;
    list->len = 0;
    list->cap = 0;
}
void token_plist_clear(token_list_t *list) {
    if (list == nullptr) return;
    token_plist_resize(list, 0);
}
void token_plist_free(const token_list_t *list) {
    if (list == nullptr || list->tokens == nullptr) return;
    free(list->tokens);
}

void token_plist_set(token_list_t *list, const token_list_t *src) {
    if (list == nullptr) return;
    if (src == nullptr) {
        token_plist_clear(list);
        return;
    }

    token_plist_resize(list, src->len);
    for (uint64_t i = 0; i < src->len; ++i)
        list->tokens[i] = src->tokens[i];
}
void token_plist_addend(token_list_t *list, token_t *token) {
    if (list == nullptr) return;
    const uint64_t len = list->len;
    token_plist_resize(list, len + 1);
    list->tokens[len] = token;
}
void token_plist_pop(token_list_t *list) {
    if (list == nullptr) return;
    const uint64_t len = list->len - 1;

    list->tokens[len] = nullptr;
    list->len = len;
}