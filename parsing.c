#include <math.h>

#include "parser.h"
#include "token.h"
#include "token.h"


#define SN_Success       0
#define SN_Nothing       1
#define SN_EOF           2
#define SN_Error         3

#define analyze_start \
const uint64_t nodes = parser->nodes.len;   \
const uint64_t pos = parser->pos;           \
node_t *expr_next = expr;                   \
token_t *token = NULL;                      \
uint8_t result = SN_Nothing, sub_result;

#define analyze_end end: \
if (result == SN_Success) return SN_Success;    \
node_list_resize(&parser->nodes, nodes);        \
node_clear(expr); parser->pos = pos;            \
return result;                                  \
analyze_end_sub


#define parser_error(msg) \
error_set_line(&parser->error, parser->tokens.tokens[parser->pos]->line); \
error_set_msg(&parser->error, msg);

#define analyze_end_sub \
sub:    result = sub_result; goto end;                                  \
err:    result = SN_Error;  parser_error("Unexpected Error") goto end;

#define expr_cast \
expr_next = node_list_append(&parser->nodes);   \
node_set(expr_next, expr); node_clear(expr);    \
node_plist_addend(&expr->nodes, expr_next);     \

#define expr_add \
expr_next = node_list_append(&parser->nodes);   \
node_plist_addend(&expr->nodes, expr_next);     \

#define expr_rem \
node_list_pop(&parser->nodes);                  \
node_plist_pop(&expr->nodes);                   \

#define check_call(call) \
sub_result = call;                              \
if (sub_result != SN_Nothing) {                 \
    if (sub_result != SN_Success) goto sub;     \
} else


#define parser_end if (parser->tokens.len <= parser->pos)
#define parser_get token = parser->tokens.tokens[parser->pos]

typedef struct {
    token_list_t tokens;
    node_list_t nodes;

    error_t error;
    uint64_t pos;
} node_parser_t;


uint8_t numb_expr(node_parser_t *parser, node_t *expr) {
    parser_end return SN_Nothing;

    token_t *parser_get;
    if (token->type != TokenType_Number) return SN_Nothing;
    parser->pos++;

    expr->type = AST_Type_Number;
    token_plist_addend(&expr->tokens, token);

    return SN_Success;
}
uint8_t geti_expr(node_parser_t *parser, node_t *expr) {
    parser_end return SN_Nothing;

    token_t *parser_get;
    if (token->type != TokenType_Identifier) return SN_Nothing;
    parser->pos++;

    expr->type = AST_Type_GetIdent;
    token_plist_addend(&expr->tokens, token);

    return SN_Success;
}
uint8_t seti_expr(node_parser_t *parser, node_t *expr) {
    parser_end return SN_Nothing;

    token_t *parser_get;
    if (token->type != TokenType_Identifier) return SN_Nothing;
    parser->pos++;

    expr->type = AST_Type_SetIdent;
    token_plist_addend(&expr->tokens, token);

    return SN_Success;
}

uint8_t math_expr(node_parser_t *parser, node_t *expr);
uint8_t rule_expr(node_parser_t *parser, node_t *expr);
uint8_t sing_expr(node_parser_t *parser, node_t *expr);

uint8_t nest_expr(node_parser_t *parser, node_t *expr, const uint32_t type) {
    analyze_start

    parser_end goto end;
    parser_get;
    if (token->type != TokenType_Special || token->sub_type != type) goto end;
    parser->pos++;

    check_call(math_expr(parser, expr)) goto err;

    parser_end goto err;
    parser_get;
    if (token->type != TokenType_Special || token->sub_type != type + 1) goto err;
    parser->pos++;

    result = SN_Success;

    analyze_end
}
uint8_t atom_expr(node_parser_t *parser, node_t *expr) {
    uint8_t result;
    if ((result = geti_expr(parser, expr)) != SN_Nothing) return result;
    if ((result = numb_expr(parser, expr)) != SN_Nothing) return result;
    return nest_expr(parser, expr, Special_LCB);
}
uint8_t qbit_expr(node_parser_t *parser, node_t *expr) {
    analyze_start

    parser_end goto end;
    parser_get;
    if (token->type != TokenType_Identifier || token->data.data[0] != 'q') goto end;
    parser->pos++;

    parser_end goto err;
    parser_get;
    if (token->type != TokenType_Special || token->sub_type != Special_UNDERSCORE) goto err;
    parser->pos++;

    expr_cast
    check_call(atom_expr(parser, expr_next)) goto err;

    expr->type = AST_Type_QBit;
    result = SN_Success;

    analyze_end
}
uint8_t prim_expr(node_parser_t *parser, node_t *expr) {
    uint8_t result;
    if ((result = qbit_expr(parser, expr)) != SN_Nothing) return result;
    if ((result = numb_expr(parser, expr)) != SN_Nothing) return result;
    if ((result = nest_expr(parser, expr, Special_LCB)) != SN_Nothing) return result;
    if ((result = nest_expr(parser, expr, Special_LSB)) != SN_Nothing) return result;
    if ((result = nest_expr(parser, expr, Special_LSQB)) != SN_Nothing) return result;
    return geti_expr(parser, expr);
}
uint8_t powr_expr(node_parser_t *parser, node_t *expr) {
    analyze_start

    check_call(prim_expr(parser, expr)) goto end;
    result = SN_Success;

    parser_end goto end;
    parser_get;
    if (token->type != TokenType_Special || token->sub_type != Special_CARET) goto end;
    parser->pos++;

    expr_cast
    expr_add
    check_call(atom_expr(parser, expr_next)) goto err;

    expr->type = AST_Type_Power;

    analyze_end
}
uint8_t sum_expr (node_parser_t *parser, node_t *expr) {
    analyze_start

    parser_end goto end;
    parser_get;
    if (token->type != TokenType_Command || token->sub_type != Command_SUM) goto end;
    parser->pos++;


    parser_end goto err;
    parser_get;
    if (token->type != TokenType_Special || token->sub_type != Special_UNDERSCORE) goto err;
    parser->pos++;

    parser_end goto err;
    parser_get;
    if (token->type != TokenType_Special || token->sub_type != Special_LCB) goto err;
    parser->pos++;

    expr_cast
    check_call(rule_expr(parser, expr_next)) goto err;

    parser_end goto err;
    parser_get;
    if (token->type != TokenType_Special || token->sub_type != Special_RCB) goto err;
    parser->pos++;

    parser_end goto err;
    parser_get;
    if (token->type != TokenType_Special || token->sub_type != Special_CARET) goto err;
    parser->pos++;

    expr_add
    check_call(atom_expr(parser, expr_next)) goto err;

    expr_add
    check_call(sing_expr(parser, expr_next)) goto err;

    expr->type = AST_Type_Sum;
    result = SN_Success;

    analyze_end
}

uint8_t main_expr(node_parser_t *parser, node_t *expr) {
    uint8_t result;
    if ((result = sum_expr(parser, expr)) != SN_Nothing) return result;
    return powr_expr(parser, expr);
}
uint8_t impl_expr(node_parser_t *parser, node_t *expr) {
    analyze_start

    expr_cast
    while (1) {
        check_call(main_expr(parser, expr_next)) break;
        expr_add
    }

    expr_rem

    if (expr->nodes.len < 2) goto end;

    expr->type = AST_Type_Implicit;
    result = SN_Success;

    analyze_end
}
uint8_t sing_expr(node_parser_t *parser, node_t *expr) {
    uint8_t result;
    if ((result = impl_expr(parser, expr)) != SN_Nothing) return result;
    return main_expr(parser, expr);
}



int8_t expr_priority(const uint16_t sub_type) {
    switch (sub_type) {
        case Special_MUL:
        case Special_DIV:
        case Special_MOD:
            return 0;
        case Special_ADD:
        case Special_SUB:
            return 1;
        default:
            return -1;
    }
}
uint8_t math_expr(node_parser_t *parser, node_t *expr) {
    analyze_start

    uint8_t stack[16];
    node_t *node[16];
    int8_t priority;
    int8_t stack_pos = -1;

    check_call(sing_expr(parser, expr)) goto end;

    while (parser->pos < parser->tokens.len) {
        parser_get;

        if (token->type != TokenType_Special || (priority = expr_priority(token->sub_type)) == -1) break;
        parser->pos++;

        node_t *prev = expr_next;
        while (stack_pos != -1 && priority > stack[stack_pos])
            prev = node[stack_pos--];

        if (stack_pos == -1 || priority < stack[stack_pos]) {
            expr_next = node_list_append(&parser->nodes);
            node_set(expr_next, prev);
            node_clear(prev);
            node_plist_addend(&prev->nodes, expr_next);

            prev->type = AST_Type_Multiplication + priority;

            ++stack_pos;
            node[stack_pos] = prev;
            stack[stack_pos] = priority;
        }

        token_plist_addend(&node[stack_pos]->tokens, token);
        expr_next = node_list_append(&parser->nodes);
        node_plist_addend(&node[stack_pos]->nodes, expr_next);

        parser_end goto err;
        check_call(sing_expr(parser, expr_next)) goto err;
    }
    result = SN_Success;
    analyze_end
}
uint8_t rule_expr(node_parser_t *parser, node_t *expr) {
    analyze_start

    expr_cast
    check_call(seti_expr(parser, expr_next)) goto err;


    parser_end goto end;
    parser_get;
    if (
        (token->type != TokenType_Special ||
            (token->sub_type != Special_EQ && token->sub_type != Special_LESS && token->sub_type != Special_GREATER)) &&
        (token->type != TokenType_Command ||
            (token->sub_type != Command_LESS_EQ && token->sub_type != Command_GREATER_EQ))) goto err;
    parser->pos++;

    token_plist_addend(&expr->tokens, token);

    expr_add
    check_call(math_expr(parser, expr_next)) goto err;

    expr->type = AST_Type_Compare;
    result = SN_Success;

    analyze_end
}

void parser_parse_ast(parser_t *parser) {
    if (parser == NULL) return;
    node_list_clear(&parser->nodes);

    node_parser_t ast_parser;
    ast_parser.tokens = parser->tokens;
    ast_parser.nodes = parser->nodes;
    ast_parser.pos = 0;

    error_init(&ast_parser.error);

    node_t *expr = node_list_append(&parser->nodes);
    if (math_expr(&ast_parser, expr) != SN_Success) {
        node_list_clear(&parser->nodes);
        error_set(&parser->error, &ast_parser.error);
    }
}
// seti, geti, numb
/// nest := {math}
/// atom := geti | numb | nest
/// scop := (math) | [math]
/// qbit := q _ atom
/// powr := prim ^ atom
/// prim := qbit | scop | atom | powr
/// sum  := \sum_{rule}^atom
/// expr := prim | sum
// impl := expr*

/// math := impl |
///         impl * math |
///         impl / math |
///         impl + math |
///         impl - math

/// rule := seti = math |
///         seti < math |
///         seti > math |
///         seti \le math |
///         seti \ge math |



