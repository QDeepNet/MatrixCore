#ifndef MATRIXCORE_PARSER_H
#define MATRIXCORE_PARSER_H

#include "bytecode.h"
#include "token.h"
#include "error.h"
#include "node.h"

typedef struct {
    uint8_t *data;
    uint64_t size;
    uint64_t N;

    token_list_t tokens;
    error_t error;

    node_t *ast;



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


uint8_t optimize_node(node_t *root, char symbol, int64_t value);
void parser_tokenize(parser_t *parser);
void parser_parse_ast(parser_t *parser);
void parser_optimizer(parser_t *parser);
void parser_interpret(const parser_t *parser);
void print_bytecode(const bytecode_t *bytecode);

static void parser_init(parser_t *parser) {
    token_list_init(&parser->tokens);
    error_init(&parser->error);
    parser->ast = node_init();
}
static void parser_free(parser_t *parser) {
    token_list_free(&parser->tokens);
    error_free(&parser->error);
    node_free(parser->ast);
}

static void parser_run(parser_t *parser) {
    parser_tokenize(parser);
    if (parser->error.present) return;
    parser_parse_ast(parser);
    if (parser->error.present) return;
    parser_optimizer(parser);
    if (parser->error.present) return;
    parser_interpret(parser);
}

#endif //MATRIXCORE_PARSER_H
