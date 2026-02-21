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

    bytecode_list_t bytecodes;
    token_list_t tokens;
    error_t error;

    node_t *ast;
} parser_t;


uint8_t optimize_node(node_t *root, char symbol, int64_t value);
void parser_tokenize(parser_t *parser);
void parser_parse_ast(parser_t *parser);
void parser_optimizer(parser_t *parser);
void parser_interpret(parser_t *parser);
void print_bytecode(const bytecode_t *bytecode);

static void parser_init(parser_t *parser) {
    bytecode_list_init(&parser->bytecodes);
    token_list_init(&parser->tokens);
    error_init(&parser->error);
    parser->ast = node_init();
}
static void parser_free(parser_t *parser) {
    bytecode_list_free(&parser->bytecodes);
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
