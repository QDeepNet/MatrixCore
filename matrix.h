// #ifndef MATRIXCORE_MATRIX_H
// #define MATRIXCORE_MATRIX_H
// #include <stdint.h>
// #include <stdlib.h>
//
// typedef struct {
//     uint64_t n;
//     uint64_t *data; // something like this [][]
// } matrix_t;
//
// typedef struct {
//     uint64_t n;
//     uint64_t *indexes;
//     uint64_t *data;
// } submatrix_t;
//
//
//
//
// static __inline__ matrix_t *matrix_init() {
//     matrix_t *matrix = malloc(sizeof(matrix_t));
//
//     matrix->n = 0;
//     matrix->data = NULL;
//     return matrix;
// }
// static __inline__ void matrix_clear(matrix_t *matrix) {
//     if (matrix == NULL) return;
//
//     if (matrix->data) free(matrix->data);
//     matrix->n = 0;
//     matrix->data = NULL;
// }
// static __inline__ void matrix_free(matrix_t *matrix) {
//     if (matrix == NULL) return;
//
//     if (matrix->data) free(matrix->data);
//     matrix->n = 0;
//     matrix->data = NULL;
//
//     free(matrix);
// }
//
// static __inline__ void matrix_set(matrix_t *matrix, uint64_t n) {
//     if (matrix == NULL) return;
//     if (matrix->data) free(matrix->data);
//     matrix->n = n;
//     matrix->data = realloc(matrix->data, sizeof(uint64_t) * n * n);
// }
//
// #endif //MATRIXCORE_MATRIX_H