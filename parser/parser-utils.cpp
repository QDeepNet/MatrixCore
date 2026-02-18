#include "parser-utils.h"


void parser_data_init(parser_data_t *data) {
    if (data == nullptr) return;
    data->data = nullptr;
    data->size = 0;
}
void parser_data_set(parser_data_t *data, const uint8_t* str, const uint64_t size) {
    if (data == nullptr) return;
    data->data = str;
    data->size = size;
}
void parser_line_init(parser_line_t *line) {
    if (line == nullptr) return;
    line->pos = 0;
    line->line_pos = 0;
    line->line_num = 0;
}
void parser_nest_init(parser_nest_t *nest) {
    if (nest == nullptr) return;
    nest->pos = 0;
}