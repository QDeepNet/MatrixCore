#ifndef MATRIXCORE_PARSER_UTILS_H
#define MATRIXCORE_PARSER_UTILS_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

#define MaxBracketNesting 256


typedef struct {
    const
    uint8_t *data;
    uint64_t size;
} parser_data_t;

typedef struct {
    uint64_t pos;
    uint64_t line_pos;
    uint64_t line_num;
} parser_line_t;

typedef struct {
    uint8_t buf[MaxBracketNesting];
    uint64_t pos;
} parser_nest_t;

static __inline__ void parser_data_init(parser_data_t *data) {
    if (data == NULL) return;
    data->data = NULL;
    data->size = 0;
}

static __inline__ void parser_data_set(parser_data_t *data, const uint8_t* str, const uint64_t size) {
    if (data == NULL) return;
    data->data = str;
    data->size = size;
}

static __inline__ void parser_line_init(parser_line_t *line) {
    if (line == NULL) return;
    line->pos = 0;
    line->line_pos = 0;
    line->line_num = 0;
}

static __inline__ void parser_nest_init(parser_nest_t *nest) {
    if (nest == NULL) return;
    nest->pos = 0;
}

#ifdef __cplusplus
}
#endif
#endif //MATRIXCORE_PARSER_UTILS_H