#include "parser.h"

typedef struct {
    parser_data_t data;
    parser_line_t line;
    parser_nest_t nest;
    error_t error;
} token_parser_t;


#define SpaceChar(c) ((c) == ' ' || (c) == '\t' || (c) == '\n')
#define IdentifierStart(c) (((c) >= 'a' && (c) <= 'z') || ((c) >= 'A' && (c) <= 'Z') || (c) == '_')
#define IdentifierChar(c) (IdentifierStart(c) || ((c) >= '0' && (c) <= '9'))


#define CharInt(c) ((c) >= '0' && (c) <= '9')

#define KeyWordChar(c) ((c) >= 'a' && (c) <= 'z')


#define GetChar {if (pos == data.size) goto end; c = data.data[pos++];}
#define tokenize_keyword_cmp(_data, value)      \
if (memcmp(data.data + 1, (_data), kw_size) == 0) { \
    type = (value);                             \
    goto end;                                   \
}

static __inline__ void tokenize_parser_pos_cast(parser_data_t *pos, const token_parser_t *parser) {
    pos->data = parser->data.data + parser->line.pos;
    pos->size = parser->data.size - parser->line.pos;
}
static __inline__ void tokenize_parser_update_pos(token_parser_t *parser, const uint64_t pos) {
    parser->line.pos += pos;
}

uint16_t Special_OneChar(const uint8_t c1, parser_nest_t *nest, error_t *error) {
    switch (c1) {
        case '%':
            return Special_MOD;
        case '(':
            if (nest->pos == MaxBracketNesting) goto err_max;
            nest->buf[nest->pos++] = Special_LSB;
            return Special_LSB;
        case ')':
            if (nest->pos == 0) goto err_min;
            if (nest->buf[--nest->pos] != Special_LSB) goto err;
            return Special_RSB;
        case '*':
            return Special_MUL;
        case '+':
            return Special_ADD;
        case ',':
            return Special_COMMA;
        case '-':
            return Special_SUB;
        case '/':
            return Special_DIV;
        case '<':
            return Special_LESS;
        case '=':
            return Special_EQ;
        case '>':
            return Special_GREATER;
        case '[':
            if (nest->pos == MaxBracketNesting) goto err_max;
            nest->buf[nest->pos++] = Special_LSQB;
            return Special_LSQB;
        case ']':
            if (nest->pos == 0) goto err_min;
            if (nest->buf[--nest->pos] != Special_LSQB) goto err;
            return Special_RSQB;
        case '^':
            return Special_CARET;
        case '_':
            return Special_UNDERSCORE;
        case '{':
            if (nest->pos == MaxBracketNesting) goto err_max;
            nest->buf[nest->pos++] = Special_LCB;
            return Special_LCB;
        case '}':
            if (nest->pos == 0) goto err_min;
            if (nest->buf[--nest->pos] != Special_LCB) goto err;
            return Special_RCB;
        default:
            break;
    }
    return Special_None;
    err_max:
    error_set_msg(error, "Maximum nested scopes exceeded.");
    return Special_None;
    err_min:
    error_set_msg(error, "Bracket mismatch detected.");
    return Special_None;
    err:
    error_set_msg(error, "Unmatched closing bracket found.");
    return Special_None;

}

void tokenize_identifier(token_t *token, token_parser_t *parser) {
    parser_data_t data;
    tokenize_parser_pos_cast(&data, parser);

    if (!IdentifierStart(data.data[0])) return;

    token->type = TokenType_Identifier;

    token_set_data(token, data.data, 1);
    token_set_line(token, parser->line);
    tokenize_parser_update_pos(parser, 1);
}
void tokenize_integer(token_t *token, token_parser_t *parser)  {
    uint64_t pos = 0;
    parser_data_t data;
    tokenize_parser_pos_cast(&data, parser);

    if (!CharInt(data.data[pos])) return;

    uint8_t c;
    uint16_t sub_type = IntType_DEC;

    do GetChar while (CharInt(c));

    if (c == '.') {
        sub_type |= IntType_FLOAT;
        do GetChar while (CharInt(c));
    }

    end:

    token->type = TokenType_Number;
    token->sub_type = sub_type;

    token_set_data(token, data.data, pos - 1);
    token_set_line(token, parser->line);
    tokenize_parser_update_pos(parser, pos - 1);
}
void tokenize_keyword(token_t *token, token_parser_t *parser) {

    uint64_t kw_size = 0;
    parser_data_t data;
    tokenize_parser_pos_cast(&data, parser);

    if (data.data[0] != '\\') return;


    for (;kw_size + 1 < data.size; kw_size++)
        if (!KeyWordChar(data.data[kw_size + 1])) break;

    uint16_t type;
    if (kw_size == 2) {
        // Keywords length 3
        tokenize_keyword_cmp("le", Command_LESS_EQ)
        tokenize_keyword_cmp("ge", Command_GREATER_EQ)
    } else if (kw_size == 3) {
        // Keywords length 3
        tokenize_keyword_cmp("sum", Command_SUM)
    }
    return;

    end:
    token->type = TokenType_Command;
    token->sub_type = type;

    token_set_data(token, data.data, kw_size + 1);
    token_set_line(token, parser->line);
    tokenize_parser_update_pos(parser, kw_size + 1);
}
void tokenize_special(token_t *token, token_parser_t *parser) {
    parser_data_t data;
    tokenize_parser_pos_cast(&data, parser);

    uint16_t res = Special_None;

    if (0 == data.size) goto end;
    const uint8_t c1 = data.data[0];
    res = Special_OneChar(c1, &parser->nest, &parser->error);


    end:
    if (res != Special_None) {
        token->type = TokenType_Special;
        token->sub_type = res;

        token_set_line(token, parser->line);
        tokenize_parser_update_pos(parser, 1);
    }
    if (parser->error.present) error_set_line(&parser->error, parser->line);
}

void tokenize_parse(token_t *token, token_parser_t *parser) {
    tokenize_special(token, parser);
    if (token->type != TokenType_None || parser->error.present) return;
    tokenize_integer(token, parser);
    if (token->type != TokenType_None || parser->error.present) return;
    tokenize_keyword(token, parser);
    if (token->type != TokenType_None) return;
    tokenize_identifier(token, parser);
}

void parser_tokenize(parser_t *parser) {
    if (parser == NULL) return;
    token_list_clear(&parser->tokens);

    if (parser->data == NULL) return;

    token_parser_t tparser;

    parser_data_init(&tparser.data);
    parser_line_init(&tparser.line);
    parser_nest_init(&tparser.nest);
    error_init(&tparser.error);

    parser_data_set(&tparser.data, parser->data, parser->size);


    token_t *token = token_list_append(&parser->tokens);

    while (tparser.line.pos < tparser.data.size) {
        const uint8_t c = tparser.data.data[tparser.line.pos];
        if (SpaceChar(c)) {
            ++tparser.line.pos;
            if (c == '\n') {
                tparser.line.line_pos = tparser.data.size + 1;
                ++tparser.line.line_num;
            }
            continue;
        }

        tokenize_parse(token, &tparser);

        if (token->type == TokenType_None) {
            if (!tparser.error.present) {
                error_set_msg(&tparser.error, "Unrecognized token");
                error_set_line(&tparser.error, tparser.line);
            }
            goto err;
        }

        token = token_list_append(&parser->tokens);
    }
    if (tparser.nest.pos != 0) {
        error_set_msg(&tparser.error, "Unterminated scope");
        error_set_line(&tparser.error, tparser.line);
        goto err;
    }

    token_list_pop(&parser->tokens);
    return;

    err:
    token_list_clear(&parser->tokens);
    error_set(&parser->error, &tparser.error);
}