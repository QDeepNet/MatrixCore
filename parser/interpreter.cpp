#include "parser.h"

typedef struct {
    bytecode_list_t list;

    
    uint64_t stack[128];
    uint64_t size;
    
    uint16_t mark;
    uint64_t pos;
} interpret_parser;
typedef struct {
    node_t *list_p[32];
    node_t *list_n[32];
} temp_lists;

void parser_append_op(const interpret_parser *parser, const uint8_t op) {
    for (uint64_t i = parser->stack[parser->size]; i < parser->list.len; i ++)
        bytecode_addend_op(parser->list.data[i], op);
}
void parser_append_val(const interpret_parser *parser, const int64_t val) {
    for (uint64_t i = parser->stack[parser->size]; i < parser->list.len; i ++)
        bytecode_addend_val(parser->list.data[i], val);
}
void parser_append_ident(const interpret_parser *parser, const uint8_t symbol) {
    for (uint64_t i = parser->stack[parser->size]; i < parser->list.len; i ++)
        bytecode_addend_op(parser->list.data[i], parser->list.data[i]->limits.data[0]->symbol == symbol ? SET_I : SET_J);
}
void parser_append_qubit(const interpret_parser *parser, const uint8_t symbol) {
    for (uint64_t i = parser->stack[parser->size]; i < parser->list.len; i ++)
        bytecode_addend_op(parser->list.data[i], parser->list.data[i]->limits.data[0]->symbol == symbol ? SET_QI : SET_QJ);
}
void parser_append_limits(const interpret_parser *parser, const node_t *node) {
    const node_t *min_node = node->nodes.nodes[0];
    const node_t *max_node = node->nodes.nodes[1];

    if (min_node->type != AST_Type_Number || max_node->type != AST_Type_Number) {
        // TODO error;
    }

    for (uint64_t i = parser->stack[parser->size]; i < parser->list.len; i ++) {
        limit_t *limit = limit_list_append(&parser->list.data[i]->limits);
        limit->symbol = node->symbol;
        limit->min = min_node->number;
        limit->max = max_node->number;
    }
}


void parser_sum_interpret(interpret_parser *parser, node_t *node) {
    temp_lists *lists;
    uint16_t len_p = 0;
    uint16_t len_n = 0;
    
    uint16_t mark = parser->mark;

    uint64_t len = parser->list.len;
    uint64_t stack = parser->stack[parser->size];

    
    parser->mark |= node->mark;
    switch (node->type) {
        case AST_Type_Sum:
            parser_append_limits(parser, node);
            parser_sum_interpret(parser, node->nodes.nodes[2]);
            break;
        case AST_Type_Number:
            parser_append_op(parser, SET);
            parser_append_val(parser, node->number);
            break;
        case AST_Type_Identifier:
            parser_append_ident(parser, node->symbol);
            break;
        case AST_Type_QBit:
            parser_append_qubit(parser, node->symbol);
            break;
        case AST_Type_Power:
            for (int64_t i = 0; i < node->nodes.len; i ++)
                parser_sum_interpret(parser, node->nodes.nodes[i]);
                parser_append_op(parser, POW);
            break;
        case AST_Type_Negative:
            for (int64_t i = 0; i < node->nodes.len; i ++)
                parser_sum_interpret(parser, node->nodes.nodes[i]);
            parser_append_op(parser, NEG);
            break;
        case AST_Type_Multiplication:
            if (node->nodes.len <= 0) break;
            lists = static_cast<temp_lists *>(malloc(sizeof(temp_lists)));
            len_p = len_n = 0;
            parser->pos = len;
            for (int64_t i = 0; i < node->nodes.len; i ++)
                if (node->nodes.nodes[i]->operation) lists->list_p[len_p++] = node->nodes.nodes[i];
                else                                 lists->list_n[len_n++] = node->nodes.nodes[i];

            if (len_p != 0) for (uint16_t i = 0; i < len_p; i ++) {
                parser_sum_interpret(parser, lists->list_p[i]);
                if (i > 0) parser_append_op(parser, MUL);
            }
            if (len_n != 0) for (uint16_t i = 0; i < len_n; i ++) {
                parser_sum_interpret(parser, lists->list_n[i]);
                if (i > 0) parser_append_op(parser, MUL);
            }

            if (len_p == 0)         parser_append_op(parser, NMl);
            else if (len_n != 0)    parser_append_op(parser, DIV);
            
            parser->pos = parser->list.len;
            free(lists);
            break;
        case AST_Type_Addition:
            if (node->nodes.len <= 0) break;
            if (!(parser->mark & 1)) goto add2;
            lists = static_cast<temp_lists *>(malloc(sizeof(temp_lists)));
            len_p = len_n = 0;
            for (int64_t i = 0; i < node->nodes.len; i ++)
                if (node->nodes.nodes[i]->operation) lists->list_p[len_p++] = node->nodes.nodes[i];
                else                                 lists->list_n[len_n++] = node->nodes.nodes[i];

            if (len_p != 0) for (uint16_t i = 0; i < len_p; i ++) {
                parser_sum_interpret(parser, lists->list_p[i]);
                if (i > 0) parser_append_op(parser, ADD);
            }
            if (len_n != 0) for (uint16_t i = 0; i < len_n; i ++) {
                parser_sum_interpret(parser, lists->list_n[i]);
                if (i > 0) parser_append_op(parser, ADD);
            }

            if (len_p == 0)         parser_append_op(parser, NEG);
            else if (len_n != 0)    parser_append_op(parser, SUB);
            free(lists);
            break;
            add2:

            for (int64_t i = 0; i < node->nodes.len; i ++) {
                parser->stack[++parser->size] = parser->list.len;
                for (uint64_t j = stack; j < len; j ++)
                    bytecode_set(bytecode_list_append(&parser->list), parser->list.data[j]);

                parser_sum_interpret(parser, node->nodes.nodes[i]);
                --parser->size;
            }

            for (uint64_t j = stack; j < len; j ++)
                bytecode_list_delete(&parser->list, j);

            break;
        default: break;
    }
    
    
    parser->mark = mark;
}
void parser_interpret(const parser_t *parser) {
    interpret_parser _parser = {};
    bytecode_list_init(&_parser.list);
    bytecode_list_append(&_parser.list);
    _parser.stack[0] = 0;
    _parser.mark = 0;
    _parser.size = 0;


    parser_sum_interpret(&_parser, parser->ast);

    // bytecode_free(_parser.bytecode);

    for (uint64_t i = 0; i < _parser.list.len; i ++) {
        print_bytecode(_parser.list.data[i]);
    }
}