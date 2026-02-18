#ifndef MATRIXCORE_ERROR_H
#define MATRIXCORE_ERROR_H

#include "parser-utils.h"
#include <string>

typedef struct {
    std::pmr::string msg;

    parser_line_t line;
    uint8_t present;
} error_t;

void error_init(error_t *error);
void error_clear(error_t *error);
void error_free(error_t *error);

void error_set(error_t *error, const error_t *src);
void error_set_msg(error_t *error, const char *msg);
void error_set_line(error_t *error, const parser_line_t &line);

#endif //MATRIXCORE_ERROR_H
