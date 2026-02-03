#include "node.h"
#include <cstdlib>

node_t *node_init() {
    const auto node = static_cast<node_t *>(malloc(sizeof(node_t)));

    node->type = AST_Type_None;

    token_plist_init(&node->tokens);
    node_plist_init(&node->nodes);

    return node;
}
void node_clear(node_t *node) {
    if (node == nullptr) return;
    node->type = AST_Type_None;

    token_plist_clear(&node->tokens);
    node_plist_clear(&node->nodes);

}
void node_free(node_t *node) {
    if (node == nullptr) return;
    token_plist_free(&node->tokens);
    node_plist_free(&node->nodes);

    free(node);
}

void node_set(node_t *node, const node_t *src) {
    if (node == nullptr) return;
    if (src == nullptr) return node_clear(node);
    node->type = src->type;

    token_plist_set(&node->tokens, &src->tokens);
    node_plist_set(&node->nodes, &src->nodes);
}


void node_list_resize(node_list_t *list, const uint64_t cap) {
    if (list->nodes == nullptr && cap != 0) {
        list->cap = cap;
        list->nodes = static_cast<node_t **>(malloc(sizeof(node_t *) * cap));
        for (uint64_t i = 0; i < cap; i++) list->nodes[i] = nullptr;
    } else if (list->cap < cap) {
        list->nodes = static_cast<node_t **>(realloc(list->nodes, sizeof(node_t *) * cap * 2));
        for (uint64_t i = list->cap, l = cap * 2; i < l; i++) list->nodes[i] = nullptr;
        list->cap = cap * 2;
    }

    for (uint64_t i = cap, l = list->len; i < l; i++) {
        if (list->nodes[i] != nullptr) node_free(list->nodes[i]);
        list->nodes[i] = nullptr;
    }
    list->len = cap;
}

void node_list_init(node_list_t *list) {
    if (list == nullptr) return;
    list->nodes = nullptr;
    list->len = 0;
    list->cap = 0;
}
void node_list_clear(node_list_t *list) {
    if (list == nullptr) return;
    node_list_resize(list, 0);
}
void node_list_free(node_list_t *list) {
    if (list == nullptr || list->nodes == nullptr) return;
    node_list_resize(list, 0);
    free(list->nodes);
}

node_t *node_list_append(node_list_t *list) {
    if (list == nullptr) return nullptr;
    const uint64_t len = list->len;
    node_list_resize(list, len + 1);
    return list->nodes[len] = node_init();
}
void node_list_pop(node_list_t *list) {
    if (list == nullptr) return;
    const uint64_t len = list->len - 1;

    if (list->nodes[len] != nullptr) node_free(list->nodes[len]);
    list->nodes[len] = nullptr;
    list->len = len;
}


void node_plist_resize(node_list_t *list, const uint64_t cap) {
    if (list->nodes == nullptr && cap != 0) {
        list->cap = cap;
        list->nodes = static_cast<node_t **>(malloc(sizeof(node_t *) * cap));
        for (uint64_t i = 0; i < cap; i++) list->nodes[i] = nullptr;
    } else if (list->cap < cap) {
        list->nodes = static_cast<node_t **>(realloc(list->nodes, sizeof(node_t *) * cap * 2));
        for (uint64_t i = list->cap, l = cap * 2; i < l; i++) list->nodes[i] = nullptr;
        list->cap = cap * 2;
    }
    list->len = cap;
}

void node_plist_init(node_list_t *list) {
    if (list == nullptr) return;
    list->nodes = nullptr;
    list->len = 0;
    list->cap = 0;
}
void node_plist_clear(node_list_t *list) {
    if (list == nullptr) return;
    node_plist_resize(list, 0);
}
void node_plist_free(const node_list_t *list) {
    if (list == nullptr || list->nodes == nullptr) return;
    free(list->nodes);
}

void node_plist_set(node_list_t *list, const node_list_t *src) {
    if (list == nullptr) return;
    if (src == nullptr) return node_plist_clear(list);

    node_plist_resize(list, src->len);
    for (uint64_t i = 0; i < src->len; ++i)
        list->nodes[i] = src->nodes[i];
}
void node_plist_addend(node_list_t *list, node_t *node) {
    if (list == nullptr) return;
    const uint64_t len = list->len;
    node_plist_resize(list, len + 1);
    list->nodes[len] = node;
}
void node_plist_pop(node_list_t *list) {
    if (list == nullptr) return;
    const uint64_t len = list->len - 1;

    list->nodes[len] = nullptr;
    list->len = len;
}