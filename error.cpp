#include "error.h"

#include <cstdlib>
#include <string.h>

void error_init(error_t *error) {
    if (error == nullptr) return;
    error->msg = "";

    parser_line_init(&error->line);
    error->present = 0;
}
void error_clear(error_t *error) {
    if (error == nullptr) return;
    error->msg = "";

    parser_line_init(&error->line);
    error->present = 0;
}
void error_free(error_t *error) {
    if (error == nullptr) return;
    error->msg = "";

    error->present = 0;
}

void error_set(error_t *error, const error_t *src) {
    if (error == nullptr) return;
    if (src == nullptr || src->present == 0) return error_init(error);

    error->msg = src->msg;
    error->line = src->line;
    error->present = 1;
}
void error_set_msg(error_t *error, const char *msg) {
    if (error == nullptr) return;

    error->msg = msg;
    error->present = 1;
}
void error_set_line(error_t *error, const parser_line_t &line) {
    if (error == nullptr) return;

    error->line = line;
}