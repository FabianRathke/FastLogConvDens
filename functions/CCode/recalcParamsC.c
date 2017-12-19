#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include "lapack.h"
#include <omp.h>

void recalcParamsC(double* X, double* y, int* T, int lenT, int dim, double* aOptNew, double* bOptNew) 
{
	int i;
	#pragma omp parallel num_threads(NUMCORES)
	{  
		int j,k;
		double *A = malloc((dim+1)*(dim+1)*sizeof(double));
	   	double *b = malloc((dim+1)*sizeof(double));
		int dimT = dim+1;
		ptrdiff_t n = dim+1, nrhs=1, lda = dim+1, info;
		ptrdiff_t *ipiv = malloc((dim+1)*sizeof(ptrdiff_t));
		#pragma omp for
		for (i=0; i < lenT; i++) {
			for (j=0; j < dimT; j++) {
				b[j] = -y[T[i*dimT + j]];
			}

			for (j=0; j < dimT; j++) {
				for (k=0; k < dim; k++) {
					A[j + dimT*k] = X[T[i*dimT + j]*dim + k];
				}
			}
			/* last row are ones */
			for (j=0; j < dimT; j++){
				A[dimT*dim+j] = 1;
			}

			dgesv_(&n,&nrhs,A,&lda,ipiv,b,&n,&info);

			for (j=0; j < dim; j++) {

				aOptNew[i*dim + j] = b[j];
			}
			bOptNew[i] = b[dim];
		}
		free(A); free(b); free(ipiv);
	}
}

