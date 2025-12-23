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
static void print_token(const token_t *token) {
    print_token__(token, 0);
}


static void print_token_list__(const token_list_t *list, const uint64_t size) {
    printf("LIST: ");
    if (list == NULL) printf("NULL\n");
    else {
        printf("(%lu)\n", list->len);

        for (int64_t i = 0; i < list->len; i ++) {
            PRINT_PREF PRINT_NEXT(i + 1 < list->len)
            print_token__(list->tokens[i], size + 2);
        }
    }
}
static void print_token_list(const token_list_t *list) {
    print_token_list__(list, 0);
}


int main(void) {
    parser_t parser = {};
    parser.data = "\\sum_i^N 2^iaq_i + \\sum_i^N \\sum_{i < j}^N 2^i2^jq_iq_j";
    parser.size = 55;

    parser_tokenize(&parser);

    print_token_list(&parser.tokens);
}