#ifndef MATRIXCORE_TOKENIZE_H
#define MATRIXCORE_TOKENIZE_H

#include <stdint.h>

#include "parser-utils.h"


#define TokenType_None         0x00
#define TokenType_Command      0x01
#define TokenType_Identifier   0x02
#define TokenType_Number       0x03
#define TokenType_Special      0x04



#define IntType_DEC             0x00
#define IntType_FLOAT           0x01


#define Command_SUM             0x01
#define Command_LESS_EQ         0x02
#define Command_GREATER_EQ      0x03


// Special Chars
#define Special_None        0x00

// Special Chars OP
#define Special_MOD         0x11 // %
#define Special_MUL         0x12 // *
#define Special_ADD         0x13 // +
#define Special_SUB         0x14 // -
#define Special_DIV         0x15 // /

// Special Chars Brackets
#define Special_LSB         0x31 // (
#define Special_RSB         0x32 // )
#define Special_LSQB        0x33 // [
#define Special_RSQB        0x34 // ]
#define Special_LCB         0x35 // {
#define Special_RCB         0x36 // }

// Special Chars Compare
#define Special_LESS        0x41 // <
#define Special_GREATER     0x42 // >

// Delimiters
#define Special_COMMA       0x01 // ,
#define Special_CARET       0x02 // ^
#define Special_UNDERSCORE  0x03 // _
#define Special_EQ          0x04 // =

typedef struct {
    uint16_t type;
    uint16_t sub_type;

    parser_data_t data;
    parser_line_t line;
} token_t;

typedef struct {
    token_t **tokens;
    uint64_t len;
    uint64_t cap;
} token_list_t;

token_t *token_init();
void token_clear(token_t *token);
void token_free(token_t *token);

void token_set_data(token_t *token, const uint8_t *str_data, uint64_t str_size);
void token_set_line(token_t *token, const parser_line_t &line);

void __token_list_resize(token_list_t *list, uint64_t cap);

void token_list_resize(token_list_t *list, uint64_t cap);

void token_list_init(token_list_t *list);
void token_list_clear(token_list_t *list);
void token_list_free(token_list_t *list);

token_t *token_list_append(token_list_t *list);
void token_list_pop(token_list_t *list);


void token_plist_resize(token_list_t *list, uint64_t cap);

void token_plist_init(token_list_t *list);
void token_plist_clear(token_list_t *list);
void token_plist_free(const token_list_t *list);

void token_plist_set(token_list_t *list, const token_list_t *src);
void token_plist_addend(token_list_t *list, token_t *token);
void token_plist_pop(token_list_t *list);

#endif //MATRIXCORE_TOKENIZE_H