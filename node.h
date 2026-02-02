#ifndef MATRIXCORE_NODE_H
#define MATRIXCORE_NODE_H
#ifdef __cplusplus
extern "C" {
#endif

#include "token.h"


#define AST_Type_None           0x00
#define AST_Type_Number         0x01
#define AST_Type_GetIdent       0x02
#define AST_Type_SetIdent       0x03
#define AST_Type_QBit           0x04
#define AST_Type_Power          0x05
#define AST_Type_Sum            0x06
#define AST_Type_Implicit       0x07
#define AST_Type_Multiplication 0x08
#define AST_Type_Addition       0x09
#define AST_Type_Compare        0x0a


typedef struct {
    struct node_st **nodes;
    uint64_t len;
    uint64_t cap;
} node_list_t;

typedef struct node_st {
    uint16_t type;

    token_list_t tokens;
    node_list_t nodes;
} node_t;


static __inline__ void node_plist_init(node_list_t *list);
static __inline__ void node_plist_clear(node_list_t *list);
static __inline__ void node_plist_free(node_list_t *list);

static __inline__ void node_plist_set(node_list_t *list, const node_list_t *src);


static __inline__ node_t *node_init() {
    node_t *node = (node_t *)malloc(sizeof(node_t));

    node->type = AST_Type_None;

    token_plist_init(&node->tokens);
    node_plist_init(&node->nodes);

    return node;
}
static __inline__ void node_clear(node_t *node) {
    if (node == NULL) return;
    node->type = AST_Type_None;

    token_plist_clear(&node->tokens);
    node_plist_clear(&node->nodes);

}
static __inline__ void node_free(node_t *node) {
    if (node == NULL) return;
    token_plist_free(&node->tokens);
    node_plist_free(&node->nodes);

    free(node);
}

static __inline__ void node_set(node_t *node, const node_t *src) {
    if (node == NULL) return;
    if (src == NULL) return node_clear(node);
    node->type = src->type;

    token_plist_set(&node->tokens, &src->tokens);
    node_plist_set(&node->nodes, &src->nodes);
}


static __inline__ void node_list_resize(node_list_t *list, const uint64_t cap) {
    if (list->nodes == NULL && cap != 0) {
        list->cap = cap;
        list->nodes = (node_t **)malloc(sizeof(node_t *) * cap);
        for (uint64_t i = 0; i < cap; i++) list->nodes[i] = NULL;
    } else if (list->cap < cap) {
        list->nodes = (node_t **)realloc(list->nodes, sizeof(node_t *) * cap * 2);
        for (uint64_t i = list->cap, l = cap * 2; i < l; i++) list->nodes[i] = NULL;
        list->cap = cap * 2;
    }

    for (uint64_t i = cap, l = list->len; i < l; i++) {
        if (list->nodes[i] != NULL) node_free(list->nodes[i]);
        list->nodes[i] = NULL;
    }
    list->len = cap;
}

static __inline__ void node_list_init(node_list_t *list) {
    if (list == NULL) return;
    list->nodes = NULL;
    list->len = 0;
    list->cap = 0;
}
static __inline__ void node_list_clear(node_list_t *list) {
    if (list == NULL) return;
    node_list_resize(list, 0);
}
static __inline__ void node_list_free(node_list_t *list) {
    if (list == NULL || list->nodes == NULL) return;
    node_list_resize(list, 0);
    free(list->nodes);
}

static __inline__ node_t *node_list_append(node_list_t *list) {
    if (list == NULL) return NULL;
    const uint64_t len = list->len;
    node_list_resize(list, len + 1);
    return list->nodes[len] = node_init();
}
static __inline__ void node_list_pop(node_list_t *list) {
    if (list == NULL) return;
    const uint64_t len = list->len - 1;

    if (list->nodes[len] != NULL) node_free(list->nodes[len]);
    list->nodes[len] = NULL;
    list->len = len;
}


static __inline__ void node_plist_resize(node_list_t *list, const uint64_t cap) {
    if (list->nodes == NULL && cap != 0) {
        list->cap = cap;
        list->nodes = (node_t **)malloc(sizeof(node_t *) * cap);
        for (uint64_t i = 0; i < cap; i++) list->nodes[i] = NULL;
    } else if (list->cap < cap) {
        list->nodes = (node_t **)realloc(list->nodes, sizeof(node_t *) * cap * 2);
        for (uint64_t i = list->cap, l = cap * 2; i < l; i++) list->nodes[i] = NULL;
        list->cap = cap * 2;
    }
    list->len = cap;
}

static __inline__ void node_plist_init(node_list_t *list) {
    if (list == NULL) return;
    list->nodes = NULL;
    list->len = 0;
    list->cap = 0;
}
static __inline__ void node_plist_clear(node_list_t *list) {
    if (list == NULL) return;
    node_plist_resize(list, 0);
}
static __inline__ void node_plist_free(node_list_t *list) {
    if (list == NULL || list->nodes == NULL) return;
    free(list->nodes);
}

static __inline__ void node_plist_set(node_list_t *list, const node_list_t *src) {
    if (list == NULL) return;
    if (src == NULL) return node_plist_clear(list);

    node_plist_resize(list, src->len);
    for (uint64_t i = 0; i < src->len; ++i)
        list->nodes[i] = src->nodes[i];
}
static __inline__ void node_plist_addend(node_list_t *list, node_t *node) {
    if (list == NULL) return;
    const uint64_t len = list->len;
    node_plist_resize(list, len + 1);
    list->nodes[len] = node;
}
static __inline__ void node_plist_pop(node_list_t *list) {
    if (list == NULL) return;
    const uint64_t len = list->len - 1;

    list->nodes[len] = NULL;
    list->len = len;
}


#ifdef __cplusplus
}
#endif
#endif //MATRIXCORE_NODE_H