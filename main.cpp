#include <iostream>
#include <random>

#include "parser/parser.h"


#include <stdio.h>

#include "parser/bytecode.h"


#define PRINT_PREF for(int _i = 0; _i < size; _i++) printf("%c", prefix[_i]);
#define PRINT_NEXT(expr) \
printf(expr? "  ├- " : "  └- "); \
prefix[size + 2] = expr? '|' : ' '; \
prefix[size + 1] = ' '; \
prefix[size] = ' ';

char prefix[100];
void *printing[100];
uint64_t printing_pos;


static void print_node_list__(const node_list_t *list, uint64_t size);

static void print_token__(const token_t *token, const uint64_t size) {
    if (token == NULL) {
        printf("NULL\n");
        return;
    }

    switch (token->type) {
        case TokenType_None:
            printf("NONE\n");
            return;
        case TokenType_Command:
            printf("COMMAND : \n");
            break;
        case TokenType_Identifier:
            printf("IDENTIFIER : \n");
            break;
        case TokenType_Number:
            printf("NUMBER : \n");
            break;
        case TokenType_Special:
            printf("SPECIAL : \n");
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


    printf("DATA: (%llu) ", token->data.size);
    for (uint64_t i = 0; i < token->data.size; i++) printf("%c", token->data.data[i]);
    printf("\n");
}
static void print_token_list__(const token_list_t *list, const uint64_t size) {
    printf("TOKENS: ");
    if (list == NULL) printf("NULL\n");
    else {
        printf("(%llu)\n", list->len);

        for (int64_t i = 0; i < list->len; i ++) {
            PRINT_PREF PRINT_NEXT(i + 1 < list->len)
            print_token__(list->tokens[i], size + 3);
        }
    }
}

static void print_node__(const node_t *node, const uint64_t size) {
    if (node == NULL) {
        printf("NULL\n");
        return;
    }

    switch (node->type) {
        case AST_Type_Number:
            printf("Number : \n");
            break;
        case AST_Type_Identifier:
            printf("Identifier : \n");
            break;
        case AST_Type_QBit:
            printf("QBit : \n");
            break;
        case AST_Type_Power:
            printf("Power : \n");
            break;
        case AST_Type_Negative:
            printf("Negative : \n");
        case AST_Type_Sum:
            printf("Sum : \n");
            break;
        case AST_Type_Multiplication:
            printf("Multiplication : \n");
            break;
        case AST_Type_Addition:
            printf("Addition : \n");
            break;
        case AST_Type_Compare:
            printf("Compare : \n");
            break;
        default:
            printf("None\n");
    }

    PRINT_PREF PRINT_NEXT(node->nodes.len > 0)
    printf("%hhu\n", node->operation);
    if (node->type == AST_Type_Number) {
        PRINT_PREF PRINT_NEXT(node->nodes.len > 0)
        printf("%lld\n", node->number);
    }
    if (node->type == AST_Type_Identifier || node->type == AST_Type_QBit || node->type == AST_Type_Sum) {
        PRINT_PREF PRINT_NEXT(node->nodes.len > 0)
        printf("%c\n", node->symbol);
    }
    if (node->nodes.len > 0) {
        PRINT_PREF PRINT_NEXT(0)
        print_node_list__(&node->nodes, size + 3);
    }
}
static void print_node_list__(const node_list_t *list, const uint64_t size) {
    printf("NODES: ");
    if (list == NULL) printf("NULL\n");
    else {
        printf("(%llu)\n", list->len);

        for (int64_t i = 0; i < list->len; i ++) {
            PRINT_PREF PRINT_NEXT(i + 1 < list->len)
            print_node__(list->nodes[i], size + 3);
        }
    }
}
static void print_bytecode__(const bytecode_t *bytecode, const uint64_t size) {
    printf("CODE: ");
    if (bytecode == nullptr) printf("NULL\n");
    else {
        printf("(%llu)\n", bytecode->len);

        for (uint64_t i = 0; i < bytecode->len;) {
            PRINT_PREF PRINT_NEXT(i + 1 < bytecode->len)
            printf("%d : ", bytecode->data[i]);
            switch (bytecode->data[i++]) {
                case NEG:
                    printf("NEG\n"); break;
                case ADD:
                    printf("ADD\n"); break;
                case SUB:
                    printf("SUB\n"); break;
                case MUL:
                    printf("MUL\n"); break;
                case DIV:
                    printf("DIV\n"); break;
                case MOD:
                    printf("MOD\n"); break;
                case POW:
                    printf("POW\n"); break;

                case SET:
                    printf("SET %lld\n", ((uint64_t *)(bytecode->data + i))[0]);
                    i += 8;
                    break;
                case SET_I:
                    printf("SET I\n"); break;
                case SET_J:
                    printf("SET J\n"); break;
                case SET_QJ:
                    printf("SET QJ\n"); break;

                default:
                    printf("\n"); break;
            }
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
static void print_token(const token_t *token) {
    print_token__(token, 0);
}
static void print_bytecode(const bytecode_t *bytecode) {
    print_bytecode__(bytecode, 0);
}

void parser_sum_limits(bytecode_t *bytecode, node_t *node) {
    const uint8_t count = bytecode->count++;
    bytecode->symbol[count] = node->nodes.nodes[0]->nodes.nodes[0]->symbol;
    bytecode->min[count] = node->nodes.nodes[0]->nodes.nodes[1]->number;
    bytecode->max[count] = node->nodes.nodes[1]->number;
}
// void parser_sum_interpret(const parser_t *parser) {
//     bytecode_t bytecode;
//     node_list_t listing;
//     node_list_t interpret;
//
//     bytecode_init(&bytecode);
//     node_plist_init(&listing);
//     node_plist_init(&interpret);
//
//     node_plist_addend(&listing, parser->nodes.nodes[0]);
//
//     while (listing.len != 0) {
//         node_t *node = listing.nodes[listing.len - 1];
//         node_plist_pop(&listing);
//
//         switch (node->type) {
//             case AST_Type_Sum:
//                 parser_sum_limits(&bytecode, node);
//                 node_plist_addend(&listing, node->nodes.nodes[2]);
//                 break;
//
//             case AST_Type_Number:
//             case AST_Type_Identifier:
//             case AST_Type_QBit:
//                 node_plist_addend(&interpret, node);
//                 break;
//
//             case AST_Type_Power:
//             case AST_Type_Negative:
//             case AST_Type_Multiplication:
//             case AST_Type_Addition:
//             case AST_Type_Compare:
//                 node_plist_addend(&interpret, node);
//                 if (node->nodes.len <= 0) break;
//
//                 for (int64_t i = 0; i < node->nodes.len; i ++)
//                     node_plist_addend(&listing, node->nodes.nodes[i]);
//                 break;
//             default:
//                 break;
//         }
//     }
//
//     while (interpret.len != 0) {
//         node_t *node = interpret.nodes[interpret.len - 1];
//         node_plist_pop(&interpret);
//         if (node->type == AST_Type_Number) {
//             bytecode_addend_op(&bytecode, SET);
//             bytecode_addend_val(&bytecode, node->number);
//         } else if (node->type == AST_Type_QBit) {
//             if (node->symbol == bytecode.symbol[0]) {
//                 bytecode_addend_op(&bytecode, SET);
//                 bytecode_addend_val(&bytecode, 1);
//             }
//             if (node->symbol == bytecode.symbol[1]) bytecode_addend_op(&bytecode, SET_QJ);
//         } else if (node->type == AST_Type_Identifier) {
//             if (node->symbol == bytecode.symbol[0]) bytecode_addend_op(&bytecode, SET_I);
//             if (node->symbol == bytecode.symbol[1]) bytecode_addend_op(&bytecode, SET_J);
//         } else if (node->type == AST_Type_Power) {
//             bytecode_addend_op(&bytecode, POW);
//         } else if (node->type == AST_Type_Negative) {
//             bytecode_addend_op(&bytecode, NEG);
//         // // } else if (node->type == AST_Type_Multiplication) {
//         // //     for (int i = 0; i < node->nodes.len - 1; i ++)
//         // //         bytecode_addend_op(&bytecode, MUL);
//         // // } else if (node->type == AST_Type_Division) {
//         // //     for (int i = 0; i < node->nodes.len - 1; i ++)
//         // //         bytecode_addend_op(&bytecode, DIV);
//         } else {
//         }
//     }
//
//     print_bytecode(&bytecode);
//
//     node_plist_free(&listing);
//     node_plist_free(&interpret);
//     bytecode_free(&bytecode);
// }
void parser_interpret(const parser_t *parser) {

}



int64_t gcd(int64_t a, int64_t b) {
    a = a < 0? -a : a;
    b = b < 0? -b : b;
    while (b) {
        a %= b;
        std::swap(a, b);
    }
    return a;
}

int64_t pow(int64_t a, int64_t e) {
    int64_t res = 1;

    while (e > 0) {
        if (e & 1) res *= a;
        a *= a;
        e >>= 1;
    }

    return res;
}
void optimize(parser_t* parser) {
    if (parser == nullptr) return;

    node_list_t stack1;
    node_list_t stack2;

    node_plist_init(&stack1);
    node_plist_init(&stack2);

    node_plist_addend(&stack1, &parser->ast);

    while (stack1.len) {
        node_t *node = stack1.nodes[stack1.len - 1];
        node_plist_addend(&stack2, node);
        node_plist_pop(&stack1);

        for (uint64_t i = 0; i < node->nodes.len; i++)
            node_plist_addend(&stack1, node->nodes.nodes[i]);
    }


    while (stack2.len) {
        node_t *node = stack2.nodes[stack2.len - 1];
        node_plist_pop(&stack2);

        if (node->type == AST_Type_Negative) {
            uint8_t operation = node->operation;
            node_t *subnode = node->nodes.nodes[0];
            if (subnode->type == AST_Type_Negative) {
                node_plist_pop(&node->nodes);
                node_move(node, subnode->nodes.nodes[0]);
                node_free(subnode);
            } else if (subnode->type == AST_Type_Number) {
                const int64_t num = subnode->number;
                node_clear(node);
                node->type = AST_Type_Number;
                node->number = -num;
            }
            node->operation = operation;
            continue;
        }
        if (node->type == AST_Type_Power) {
            node_t *subnode1 = node->nodes.nodes[0];
            node_t *subnode2 = node->nodes.nodes[1];

            if (subnode1->type != AST_Type_Number) continue;
            if (subnode2->type != AST_Type_Number) continue;

            const int64_t num = pow(subnode1->number, subnode2->number);
            uint8_t operation = node->operation;
            node_clear(node);
            node->type = AST_Type_Number;
            node->number = num;
            node->operation = operation;
        }

        // case : + (+) or * (*)
        if (node->type != AST_Type_Addition && node->type != AST_Type_Multiplication) continue;

        uint8_t add = node->type == AST_Type_Addition;

        int64_t pos = add? 0 : 1;
        int64_t neg = add? 0 : 1;

        for (uint64_t i = 0; i < node->nodes.len; i++) {
            node_t *subnode = node->nodes.nodes[i];

            if (subnode->type == AST_Type_Number) {
                int64_t *ptr = subnode->operation == AST_Operation_Negative? &neg : &pos;

                if (add)    *ptr += subnode->number;
                else        *ptr *= subnode->number;

                const int64_t num = add? (pos < neg? pos: neg) : gcd(pos, neg);
                if (add)    pos -= num;
                else        pos /= num;
                if (add)    neg -= num;
                else        neg /= num;

                node_list_delete(&node->nodes, i);
                i--;
            } else if (subnode->type == node->type) {
                for (uint64_t j = 0; j < subnode->nodes.len; j++) {
                    node_t *subsubnode = subnode->nodes.nodes[j];
                    subsubnode->operation ^= subnode->operation ^ 1;
                    node_plist_addend(&node->nodes, subsubnode);
                }
                node_plist_clear(&subnode->nodes);
                node_list_delete(&node->nodes, i);
                i--;
            }
        }

        if (node->nodes.len == 0) {
            uint8_t operation = node->operation;
            node_clear(node);
            node->type = AST_Type_Number;
            node->number = add? pos - neg : pos / neg;
            node->operation = operation;
        } else {
            if (pos !=(add? 0 : 1)) {
                node_t *subnode = node_list_append(&node->nodes);
                subnode->type = AST_Type_Number;
                subnode->number = pos;
                subnode->operation = AST_Operation_Positive;
            }

            if (neg != (add? 0 : 1)) {
                node_t *subnode = node_list_append(&node->nodes);
                subnode->type = AST_Type_Number;
                subnode->number = neg;
                subnode->operation = AST_Operation_Negative;
            }
        }

    }
}

#include "solver/cpu_adapter.cpp"
#include "solver/cfc.h"

int main(void) {
    // parser_t parser = {};
    // // const char *data = "\\sum_{i=0}^N \\sum_{i=0}^N (i+j + 10 + 1)q_iq_j";
    // // const char *data = "\\sum_{i=0}^{N-1} (i + 100) 10 q_i";
    // const char *data = "a + 10 - (a - 10)";
    // parser.data = (uint8_t *)data;
    // parser.size = strlen(data);
    //
    // parser_tokenize(&parser);
    // parser_parse_ast(&parser);
    // print_node(&parser.ast);
    // optimize(&parser);
    //
    // // print_token_list(&parser.tokens);
    //
    // print_node(&parser.ast);

    int N=6;
    int B=1;
    int n_iter=1000;

    float Jraw[36]={
        0,-1, 1, 0, 1,-1,
       -1, 0,-1, 1, 0, 1,
        1,-1, 0,-1, 1, 0,
        0, 1,-1, 0,-1, 1,
        1, 0, 1,-1, 0,-1,
       -1, 1, 0, 1,-1, 0
   };

    std::vector<float> J(Jraw,Jraw+36);

    CPUAdapter cpu;

    CFC solver(N,B,n_iter,J,&cpu);
    solver.run();

    std::cout<<"Spins:\n";
    for(int i=0;i<N;i++)
        std::cout<<(solver.x[i]>=0?1:-1)<<" ";
}