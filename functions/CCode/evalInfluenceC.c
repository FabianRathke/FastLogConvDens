#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <float.h>

void evalInfluenceC(double* influence, double* Y, double* a, double* b, double gamma, int M, int dim, int nH)
{
	double aGamma[dim*nH], bGamma[nH];
	int i,j,k;
	double epsCalcExp = -350; // this is the maximal allowed difference between the maximum and any other hyperplane in log-space; this could be possibly -700, as exp(-700) is roughly the realmin, but we are greedy and drop some hyperplanes
	
	for (i=0; i < nH; i++) {
		for (j=0; j < dim; j++) {
			aGamma[i+(j*nH)] = gamma*a[i+(j*nH)];
		}
		bGamma[i] = gamma*b[i];
		influence[i] = 0;
	}

	/* Evaluate the influence for grid points */
	#pragma omp parallel
	{
		double *Ytmp = calloc(dim,sizeof(double));
		double stInnerMax;
		double sum_st, sum_st_inv;
		double *st = calloc(nH,sizeof(double));
		double *stInner = calloc(nH,sizeof(double));
		double *influencePrivate = calloc(nH,sizeof(double));
		int numElements, idxElements[nH], idxSave;
		/* calculate gradient for grid points */
		#pragma omp for private(i,k)
		for (j=0; j < M; j++) {
			stInnerMax = -DBL_MAX;
			for (k=0; k < dim; k++) {
				Ytmp[k] = Y[k + j*dim];
			}
			for (i=0; i < nH; i++) {
				stInner[i] = bGamma[i] + aGamma[i]*Ytmp[0];
			}
			for (k=1; k < dim; k++) {
				for (i=0; i < nH; i++) {
					stInner[i] = stInner[i] + (aGamma[i+k*nH]*Ytmp[k]);
				}
			}

			// find maximum element
			for (i=0; i < nH; i++) {
				if (stInner[i] > stInnerMax) {
					stInnerMax = stInner[i];
				}
			}

			sum_st = 0; numElements = 0;
			// calculate st only for those entries that wont be zero
			for (i=0; i < nH; i++) {
				if (stInner[i] - stInnerMax > epsCalcExp) {
					st[numElements] = exp(stInner[i]-stInnerMax);
					idxElements[numElements] = i;
					sum_st += st[numElements++];
				}
			}
			sum_st_inv = 1/sum_st;
			for (i=0; i < numElements; i++) {
				idxSave = idxElements[i];
				influencePrivate[idxSave] += st[i]*sum_st_inv;
			}
		}
		#pragma omp critical
		{
			for (i=0; i < nH; i++) {
				influence[i] += influencePrivate[i];
			}
		}
	} // end of pragma parallel
}

