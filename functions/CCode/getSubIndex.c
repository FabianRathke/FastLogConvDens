#include <stdlib.h>
#include <stdio.h>

void getSubIndex(int dim, int M, int MPow, int* subIdx) {
    int k, j, raiseIndex;

    for (j=0; j < dim; j++) {
        subIdx[j] = 0;
    }

    /* the first index always grows until M, then is reset to 0; other indices only grows if the previous one reaches M */
    for (j=1; j < MPow; j++) {
        /* increase the first one */
        raiseIndex = 0;
        if (subIdx[(j-1)*dim] < M-1) {
            subIdx[j*dim] = subIdx[(j-1)*dim]+1;
        } else {
            subIdx[j*dim] = 0;
            raiseIndex = 1;
        }
        for (k=1; k < dim; k++) {
            if (raiseIndex) {
                if (subIdx[k+(j-1)*dim] < M-1) {
                    subIdx[k+j*dim] = subIdx[k + (j-1)*dim] + 1;
                    raiseIndex = 0;
                } else {
                    subIdx[k+j*dim] = 0;
                    raiseIndex = 1;
                }
            } else {
                subIdx[k + j*dim] = subIdx[k + (j-1)*dim];
            }
        }
    }
}


