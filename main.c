#include "parser.h"


#include <stdio.h>


#define PRINT_PREF for(int _i = 0; _i < size; _i++) printf("%c", prefix[_i]);
#define PRINT_NEXT(expr) \
printf(expr? "\t├- " : "\t└- "); \
prefix[size + 1] = expr? '|' : ' '; \
prefix[size] = '\t';

char prefix[100];
void *printing[100];
uint64_t printing_pos;


static void print_node_list__(const node_list_t *list, const uint64_t size);

static void print_token__(const token_t *token, const uint64_t size) {
    printf("TOKEN: ");
    if (token == NULL) {
        printf("NULL\n");
        return;
    }
    printf("\n");


    PRINT_PREF PRINT_NEXT(token->type != TokenType_None)

    printf("TYPE: ");
    switch (token->type) {
        case TokenType_None:
            printf("NONE\n");
            return;
        case TokenType_Command:
            printf("COMMAND\n");
            break;
        case TokenType_Identifier:
            printf("IDENTIFIER\n");
            break;
        case TokenType_Number:
            printf("NUMBER\n");
            break;
        case TokenType_Special:
            printf("SPECIAL\n");
            break;
        default: ;
    }

    PRINT_PREF PRINT_NEXT(0)
    if (token->type == TokenType_Special) {

        printf("VALUE: ");
        switch (token->sub_type) {
            case Special_MOD:
                printf("%c", '%');
                break;
            case Special_LSB:
                printf("(");
                break;
            case Special_RSB:
                printf(")");
                break;
            case Special_MUL:
                printf("*");
                break;
            case Special_ADD:
                printf("+");
                break;
            case Special_COMMA:
                printf(",");
                break;
            case Special_SUB:
                printf("-");
                break;
            case Special_DIV:
                printf("/");
                break;
            case Special_LESS:
                printf("<");
                break;
            case Special_EQ:
                printf("=");
                break;
            case Special_GREATER:
                printf(">");
                break;
            case Special_LSQB:
                printf("[");
                break;
            case Special_RSQB:
                printf("]");
                break;
            case Special_CARET:
                printf("^");
                break;
            case Special_UNDERSCORE:
                printf("_");
                break;
            case Special_LCB:
                printf("{");
                break;
            case Special_RCB:
                printf("}");
                break;
            default: ;
        }
        printf("\n");
        return;
    }


    printf("DATA: (%lu) ", token->data.size);
    for (uint64_t i = 0; i < token->data.size; i++) printf("%c", token->data.data[i]);
    printf("\n");
}
static void print_token_list__(const token_list_t *list, const uint64_t size) {
    printf("TOKENS: ");
    if (list == NULL) printf("NULL\n");
    else {
        printf("(%lu)\n", list->len);

        for (int64_t i = 0; i < list->len; i ++) {
            PRINT_PREF PRINT_NEXT(i + 1 < list->len)
            print_token__(list->tokens[i], size + 2);
        }
    }
}

static void print_node__(const node_t *node, const uint64_t size) {
    printf("NODE: ");
    if (node == NULL) {
        printf("NULL\n");
        return;
    }
    printf("\n");


    PRINT_PREF PRINT_NEXT(node->type != MainType_None)
    printf("TYPE: ");

    switch (node->type) {
        case MainType_None:
            printf("None\n");
            return;
        case MainType_Value:
            printf("Value\n");
            PRINT_PREF PRINT_NEXT(node->tokens.len > 0 || node->nodes.len > 0)
            printf("SUB TYPE: ");
            switch (node->type) {
                case ValueType_Number:
                    printf("Number\n");
                    break;
                case ValueType_GetIdent:
                    printf("GetIdent\n");
                    break;
                case ValueType_SetIdent:
                    printf("SetIdent\n");
                    break;
                default:
                    break;
            }
            break;
        case MainType_Expr:
            printf("Expr\n");
            PRINT_PREF PRINT_NEXT(node->tokens.len > 0 || node->nodes.len > 0)
            printf("SUB TYPE: ");

            switch (node->type) {
                case ExprType_QBit:
                    printf("QBit\n");
                    break;
                case ExprType_Power:
                    printf("Power\n");
                    break;
                case ExprType_Sum:
                    printf("Sum\n");
                    break;
                case ExprType_Implicit:
                    printf("Implicit\n");
                    break;
                default:
                    break;
            }
            break;
        case MainType_Oper:
            printf("Oper\n");
            PRINT_PREF PRINT_NEXT(node->tokens.len > 0 || node->nodes.len > 0)
            printf("SUB TYPE: ");
            switch (node->type) {
                case OperType_Multiplication:
                    printf("Multiplication\n");
                    break;
                case OperType_Addition:
                    printf("Addition\n");
                    break;
                case OperType_Compare:
                    printf("Compare\n");
                    break;
                default:
                    break;
            }
            break;
        default: ;
    }

    if (node->tokens.len > 0) {
        PRINT_PREF PRINT_NEXT(node->nodes.len > 0)
        print_token_list__(&node->tokens, size + 2);
    }
    if (node->nodes.len > 0) {
        PRINT_PREF PRINT_NEXT(0)
        print_node_list__(&node->nodes, size + 2);
    }
}
static void print_node_list__(const node_list_t *list, const uint64_t size) {
    printf("NODES: ");
    if (list == NULL) printf("NULL\n");
    else {
        printf("(%lu)\n", list->len);

        for (int64_t i = 0; i < list->len; i ++) {
            PRINT_PREF PRINT_NEXT(i + 1 < list->len)
            print_node__(list->nodes[i], size + 2);
        }
    }
}


static void print_token_list(const token_list_t *list) {
    print_token_list__(list, 0);
}
static void print_node_list(const node_list_t *list) {
    print_node_list__(list, 0);
}
static void print_node(const node_t *node) {
    print_node__(node, 0);
}


int main(void) {
    parser_t parser = {};
    //\\sum_i^N 2^iaq_i + \\sum_i^N \\sum_{i < j}^N 2^i2^jq_iq_j

    char *data = "1 + 2";
    parser.data = (uint8_t *)data;
    parser.size = strlen(data);

    parser_tokenize(&parser);
    parser_parse_ast(&parser);

    // print_token_list(&parser.tokens);

    print_node_list(&parser.nodes);
}