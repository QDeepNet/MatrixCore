#ifndef MATRIXCORE_ERROR_H
#define MATRIXCORE_ERROR_H

#include <stdlib.h>
#include <string.h>

#include "parser-utils.h"

typedef struct {
    char *msg;

    parser_line_t line;
    uint8_t present;
} error_t;

static __inline__ void error_init(error_t *error) {
    if (error == NULL) return;
    error->msg = NULL;

    parser_line_init(&error->line);
    error->present = 0;
}
static __inline__ void error_clear(error_t *error) {
    if (error == NULL) return;
    if (error->msg) free(error->msg);
    error->msg = NULL;

    parser_line_init(&error->line);
    error->present = 0;
}
static __inline__ void error_free(error_t *error) {
    if (error == NULL) return;
    if (error->msg) free(error->msg);
    error->msg = NULL;

    error->present = 0;
}

static __inline__ void error_set(error_t *error, const error_t *src) {
    if (error == NULL) return;
    if (src == NULL || src->present == 0) return error_init(error);

    error->msg = strdup(src->msg);
    error->line = src->line;
    error->present = 1;
}
static __inline__ void error_set_msg(error_t *error, const char *msg) {
    if (error == NULL) return;

    error->msg = strdup(msg);
    error->present = 1;
}
static __inline__ void error_set_line(error_t *error, const parser_line_t line) {
    if (error == NULL) return;

    error->line = line;
}

#endif //MATRIXCORE_ERROR_H
