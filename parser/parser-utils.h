#ifndef MATRIXCORE_PARSER_UTILS_H
#define MATRIXCORE_PARSER_UTILS_H

#include <cstdint>

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


static void parser_data_set(parser_data_t *data, const uint8_t* str, const uint64_t size) {
    if (data == nullptr) return;
    data->data = str;
    data->size = size;
}


#endif //MATRIXCORE_PARSER_UTILS_H