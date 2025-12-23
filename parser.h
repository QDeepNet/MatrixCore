#ifndef MATRIXCORE_PARSER_H
#define MATRIXCORE_PARSER_H


#include "parser-utils.h"
#include "token.h"
#include "error.h"
#include "node.h"

typedef struct {
    uint8_t *data;
    uint64_t size;

    token_list_t tokens;
    node_list_t nodes;
    error_t error;



    //     struct node_list_st nodes;
    //     struct token_list_st tokens;
    //     struct bytecode_list_st codes; //
    //     struct closure_list_st closures; //
    //     struct variable_list_list_st variables; //
    //
    //     struct bytecode_list_st codes_stack;
    //     struct closure_list_st closures_stack;
    //     struct variable_list_list_st variables_stack;

    // struct list_st *const_objects; //
    // struct list_st *temp_stack;
    // struct list_st *var_stack; //
    // uint64_t var_start_pos;
} parser_t;


void parser_tokenize(parser_t *parser);
void parser_parse_ast(parser_t *parser);

#endif //MATRIXCORE_PARSER_H
