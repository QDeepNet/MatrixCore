#ifndef MATRIXCORE_TOKENIZE_H
#define MATRIXCORE_TOKENIZE_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>

#include "parser-utils.h"


#define TokenType_None         0x00
#define TokenType_Command      0x01
#define TokenType_Identifier   0x02
#define TokenType_Number       0x03
#define TokenType_Special      0x04



#define IntType_DEC             0x00
#define IntType_FLOAT           0x01


#define Command_SUM             0x01
#define Command_LESS_EQ         0x02
#define Command_GREATER_EQ      0x03


// Special Chars
#define Special_None        0x00

// Special Chars OP
#define Special_MOD         0x11 // %
#define Special_MUL         0x12 // *
#define Special_ADD         0x13 // +
#define Special_SUB         0x14 // -
#define Special_DIV         0x15 // /

// Special Chars Brackets
#define Special_LSB         0x31 // (
#define Special_RSB         0x32 // )
#define Special_LSQB        0x33 // [
#define Special_RSQB        0x34 // ]
#define Special_LCB         0x35 // {
#define Special_RCB         0x36 // }

// Special Chars Compare
#define Special_LESS        0x41 // <
#define Special_GREATER     0x42 // >

// Delimiters
#define Special_COMMA       0x01 // ,
#define Special_CARET       0x02 // ^
#define Special_UNDERSCORE  0x03 // _
#define Special_EQ          0x04 // =

typedef struct {
    uint16_t type;
    uint16_t sub_type;

    parser_data_t data;
    parser_line_t line;
} token_t;

typedef struct {
    token_t **tokens;
    uint64_t len;
    uint64_t cap;
} token_list_t;

static __inline__ token_t *token_init() {
    token_t *token = (token_t *)malloc(sizeof(token_t));
    token->type = TokenType_None;
    token->sub_type = TokenType_None;

    parser_data_init(&token->data);
    parser_line_init(&token->line);
    return token;
}
static __inline__ void token_clear(token_t *token) {
    if (token == NULL) return;
    token->type = TokenType_None;
    token->sub_type = TokenType_None;

    parser_data_init(&token->data);
    parser_line_init(&token->line);
}
static __inline__ void token_free(token_t *token) {
    if (token == NULL) return;
    free(token);
}

static __inline__ void token_set_data(token_t *token, const uint8_t *str_data, const uint64_t str_size) {
    if (token == NULL) return;
    token->data.data = str_data;
    token->data.size = str_size;
}
static __inline__ void token_set_line(token_t *token, const parser_line_t line) {
    if (token == NULL) return;
    token->line = line;
}

static __inline__ void __token_list_resize(token_list_t *list, const uint64_t cap) {
    if (list->tokens == NULL && cap != 0) {
        list->cap = cap;
        list->tokens = (token_t **)malloc(sizeof(token_t *) * cap);
        for (uint64_t i = 0; i < cap; i++) list->tokens[i] = NULL;
    } else if (list->cap < cap) {
        list->tokens = (token_t **)realloc(list->tokens, sizeof(token_t *) * cap * 2);
        for (uint64_t i = list->cap, l = cap * 2; i < l; i++) list->tokens[i] = NULL;
        list->cap = cap * 2;
    }
}

static __inline__ void token_list_resize(token_list_t *list, const uint64_t cap) {
    __token_list_resize(list, cap);

    for (uint64_t i = cap, l = list->len; i < l; i++) {
        if (list->tokens[i] != NULL) token_free(list->tokens[i]);
        list->tokens[i] = NULL;
    }
    list->len = cap;
}

static __inline__ void token_list_init(token_list_t *list) {
    if (list == NULL) return;
    list->tokens = NULL;
    list->len = 0;
    list->cap = 0;
}
static __inline__ void token_list_clear(token_list_t *list) {
    if (list == NULL) return;
    token_list_resize(list, 0);
}
static __inline__ void token_list_free(token_list_t *list) {
    if (list == NULL || list->tokens == NULL) return;
    token_list_resize(list, 0);
    free(list->tokens);
}

static __inline__ token_t *token_list_append(token_list_t *list) {
    if (list == NULL) return NULL;
    const uint64_t len = list->len;
    token_list_resize(list, len + 1);
    return list->tokens[len] = token_init();
}
static __inline__ void token_list_pop(token_list_t *list) {
    if (list == NULL) return;
    const uint64_t len = list->len - 1;

    if (list->tokens[len] != NULL) token_free(list->tokens[len]);
    list->tokens[len] = NULL;
    list->len = len;
}


static __inline__ void token_plist_resize(token_list_t *list, const uint64_t cap) {
    __token_list_resize(list, cap);
    list->len = cap;
}

static __inline__ void token_plist_init(token_list_t *list) {
    if (list == NULL) return;
    list->tokens = NULL;
    list->len = 0;
    list->cap = 0;
}
static __inline__ void token_plist_clear(token_list_t *list) {
    if (list == NULL) return;
    token_plist_resize(list, 0);
}
static __inline__ void token_plist_free(token_list_t *list) {
    if (list == NULL || list->tokens == NULL) return;
    free(list->tokens);
}

static __inline__ void token_plist_set(token_list_t *list, const token_list_t *src) {
    if (list == NULL) return;
    if (src == NULL) return token_plist_clear(list);

    token_plist_resize(list, src->len);
    for (uint64_t i = 0; i < src->len; ++i)
        list->tokens[i] = src->tokens[i];
}
static __inline__ void token_plist_addend(token_list_t *list, token_t *token) {
    if (list == NULL) return;
    const uint64_t len = list->len;
    token_plist_resize(list, len + 1);
    list->tokens[len] = token;
}
static __inline__ void token_plist_pop(token_list_t *list) {
    if (list == NULL) return;
    const uint64_t len = list->len - 1;

    list->tokens[len] = NULL;
    list->len = len;
}

#ifdef __cplusplus
}
#endif
#endif //MATRIXCORE_TOKENIZE_H