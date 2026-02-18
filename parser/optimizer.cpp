#include "parser.h"


int64_t gcd(int64_t a, int64_t b) {
    a = a < 0 ? -a : a;
    b = b < 0 ? -b : b;
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

void parser_optimizer(parser_t *parser) {
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

        const uint8_t operation = node->operation;
        if (node->type == AST_Type_Identifier && node->symbol == 'N') {
            node_clear(node);
            node->type = AST_Type_Number;
            node->number = parser->N;
            node->operation = operation;
            continue;
        }

        if (node->type == AST_Type_Negative) {
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
            const node_t *subnode1 = node->nodes.nodes[0];
            const node_t *subnode2 = node->nodes.nodes[1];

            if (subnode1->type != AST_Type_Number) continue;
            if (subnode2->type != AST_Type_Number) continue;

            const int64_t num = pow(subnode1->number, subnode2->number);
            node_clear(node);
            node->type = AST_Type_Number;
            node->number = num;
            node->operation = operation;
        }

        if (node->type == AST_Type_Sum) {

            const node_t *subnode1 = node->nodes.nodes[0];
            const node_t *subnode2 = node->nodes.nodes[1];

            if (subnode1->type != AST_Type_Number || subnode2->type != AST_Type_Number) {
                // error;
            }

            node->limit_min = subnode1->number;
            node->limit_max = subnode2->number;

            node_list_delete(&node->nodes, 0);
            node_list_delete(&node->nodes, 2);
        }

        // case : + (+) or * (*)
        if (node->type != AST_Type_Addition && node->type != AST_Type_Multiplication) continue;

        uint8_t add = node->type == AST_Type_Addition;

        int64_t pos = add ? 0 : 1;
        int64_t neg = add ? 0 : 1;

        for (uint64_t i = 0; i < node->nodes.len; i++) {
            node_t *subnode = node->nodes.nodes[i];

            if (subnode->type == AST_Type_Number) {
                int64_t *ptr = subnode->operation == AST_Operation_Negative ? &neg : &pos;

                if (add) *ptr += subnode->number;
                else *ptr *= subnode->number;

                const int64_t num = add ? (pos < neg ? pos : neg) : gcd(pos, neg);
                if (add) pos -= num;
                else pos /= num;
                if (add) neg -= num;
                else neg /= num;

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
            node_clear(node);
            node->type = AST_Type_Number;
            node->number = add ? pos - neg : pos / neg;
            node->operation = operation;
        } else {
            if (pos != (add ? 0 : 1)) {
                node_t *subnode = node_list_append(&node->nodes);
                subnode->type = AST_Type_Number;
                subnode->number = pos;
                subnode->operation = AST_Operation_Positive;
            }

            if (neg != (add ? 0 : 1)) {
                node_t *subnode = node_list_append(&node->nodes);
                subnode->type = AST_Type_Number;
                subnode->number = neg;
                subnode->operation = AST_Operation_Negative;
            }
        }
    }
}
