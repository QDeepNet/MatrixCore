#ifndef MATRIXCORE_NODE_H
#define MATRIXCORE_NODE_H

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


node_t *node_init();
void node_clear(node_t *node);
void node_free(node_t *node);

void node_set(node_t *node, const node_t *src);


void node_list_resize(node_list_t *list, uint64_t cap);

void node_list_init(node_list_t *list);
void node_list_clear(node_list_t *list);
void node_list_free(node_list_t *list);

node_t *node_list_append(node_list_t *list);
void node_list_pop(node_list_t *list);


void node_plist_resize(node_list_t *list, uint64_t cap);

void node_plist_init(node_list_t *list);
void node_plist_clear(node_list_t *list);
void node_plist_free(const node_list_t *list);

void node_plist_set(node_list_t *list, const node_list_t *src);
void node_plist_addend(node_list_t *list, node_t *node);
void node_plist_pop(node_list_t *list);


#endif //MATRIXCORE_NODE_H