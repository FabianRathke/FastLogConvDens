#include <stdlib.h>
#include <stdio.h>

void preCalcGrid(int dim, int M, int lenVec, double* vec, double* subGrid, double* preCalcGrid) {
    int i,j,k;

    for (j=0; j < dim; j++) {
        for (i=0; i < M; i++) {
            for (k=0; k < lenVec; k++) {
                preCalcGrid[i + j*M + k*dim*M] = vec[k + j*lenVec]*subGrid[i + j*M];
            }
        }
    }
}

