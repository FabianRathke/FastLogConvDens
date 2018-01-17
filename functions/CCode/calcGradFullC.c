#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <float.h>

void calcGradFullC(double* gradA, double* gradB, double* influence, double* TermA, double* TermB, double* X, double* XW, double* grid, unsigned short int* YIdx, double* a, double* b, double gamma, double weight, double* delta, int N, int M, int dim, int nH, int n)
{
	double *grad_st_tmp = calloc(nH*(dim+1),sizeof(double));
	double *aGamma = malloc(dim*nH*sizeof(double)); 
	double *bGamma = malloc(nH*sizeof(double));
	int dimnH = dim*nH;
	int i,j,k;
	double factor = 1/gamma;
	double *gridLocal = malloc(dim*sizeof(double));	
	double epsCalcExp = -25; /* this is the maximal difference between the maximum and any other hyperplane in log-space; increase this value for more accuracy */
	double TermALocal, TermBLocal;
	
	/* initialize some variables */
	for (k=0; k < dim; k++) {
		gridLocal[k] = grid[k];
	}

	for (i=0; i < nH; i++) {
		for (j=0; j < dim; j++) {
			aGamma[i+(j*nH)] = gamma*a[i+(j*nH)];
		}
		bGamma[i] = gamma*b[i];
		influence[i] = 0;
	}

	for (i=0; i < nH*(dim+1); i++) {
		gradA[i] = 0; gradB[i] = 0;
	}

	/* calculate gradient for samples */ 
	TermALocal = 0;	*TermA = 0;
	#pragma omp parallel num_threads(NUMCORES)
    {
      	/* const int nthreads = omp_get_num_threads();
		 * printf("Number of threads: %d\n",nthreads); */
   	  	double ftInnerMax;
        double sum_ft, sum_ft_inv;
        double *ft = calloc(nH,sizeof(double));
        double *ftInner = calloc(nH,sizeof(double));
        double *grad_ft_private = calloc(nH*(dim+1),sizeof(double));
		int *idxElements = malloc(nH*sizeof(int));
		int numElements, idxSave;
        double ftTmp;

        /* calculate gradient for samples X */
        #pragma omp for schedule(dynamic) private(i,k) reduction(+:TermALocal)
        for (j=0; j < N; j++) {
            ftInnerMax = -DBL_MAX; numElements = 0;
            for (i=0; i < nH; i++) {
                ftTmp = bGamma[i] + aGamma[i]*X[j];
                for (k=1; k < dim; k++) {
                    ftTmp += aGamma[i+k*nH]*X[j + (k*N)];
                }
                if (ftTmp > ftInnerMax - 25) {
                    if (ftTmp > ftInnerMax) {
                        ftInnerMax = ftTmp;
                    }
                    ftInner[numElements] = ftTmp;
                    idxElements[numElements++] = i;
                }
            }

            sum_ft = 0;
            /* calculate ft only for those entries that will be non-zero */
            for (i=0; i < numElements; i++) {
                ft[i] = exp(ftInner[i]-ftInnerMax);
                sum_ft += ft[i];
            }

            TermALocal += XW[j]*(ftInnerMax + log(sum_ft))*factor;
            sum_ft_inv = 1/sum_ft*XW[j];

            /* update the gradient */
            for (i=0; i < numElements; i++) {
                idxSave = idxElements[i];
                for (k=0; k < dim; k++) {
                    grad_ft_private[idxSave + (k*nH)] += ft[i]*X[j+(k*N)]*sum_ft_inv;
                }
                grad_ft_private[idxSave + (dim*nH)] += ft[i]*sum_ft_inv;
            }
        }
		#pragma omp critical
        {
			for (i=0; i < nH*(dim+1); i++) {
            	gradA[i] += grad_ft_private[i];
            }
        }
		free(ft); free(ftInner); free(grad_ft_private); free(idxElements);
	}
	*TermA = TermALocal;

	/* Calculate gradient for grid points */ 
	TermBLocal = 0; *TermB = 0;
	#pragma omp parallel num_threads(NUMCORES)
	{
		double *Ytmp = calloc(dim,sizeof(double));
		double stInnerMax; double stInnerCorrection = 0;
		double sum_st, sum_st_inv, tmpVal, sum_st_inv2;
		double *st = calloc(nH,sizeof(double));
		double *stInner = calloc(nH,sizeof(double));
		double *grad_st_private = calloc(nH*(dim+1),sizeof(double));
		double *influencePrivate = calloc(nH,sizeof(double));
		int *idxElements = malloc(nH*sizeof(int));
		int numElements, idxSave;
		double stTmp;
		/* calculate gradient for grid points */
		#pragma omp for schedule(dynamic) private(i,k) reduction(+:TermBLocal)
		for (j=0; j < M; j++) {
			stInnerMax = -DBL_MAX; numElements = 0;
			for (k=0; k < dim; k++) {
				Ytmp[k] = gridLocal[k]+delta[k]*YIdx[k + j*dim];
			}
            for (i=0; i < nH; i++) {
                stTmp = bGamma[i] + aGamma[i]*Ytmp[0];
                for (k=1; k < dim; k++) {
                    stTmp += (aGamma[i+k*nH]*Ytmp[k]);
                }
                if (stTmp > stInnerMax - 25) {
                    if (stTmp > stInnerMax) {
                        stInnerMax = stTmp;
                    }
                    stInner[numElements] = stTmp;
                    idxElements[numElements++] = i;
                }
            }

            sum_st = 0;
            /* calculate st only for those entries that wont be zero: exp^(-large number) */
            for (i=0; i < numElements; i++) {
                st[i] = exp(stInner[i]-stInnerMax);
                sum_st += st[i];
            }
			
			stInnerCorrection = exp(-stInnerMax*factor);
			tmpVal = pow(sum_st,-factor)*stInnerCorrection;
			
			TermBLocal += tmpVal; /*evalGrid[j] = tmpVal; */
			sum_st_inv2 = 1/sum_st;
			sum_st_inv = tmpVal*sum_st_inv2;
																		
			for (i=0; i < numElements; i++) {
				idxSave = idxElements[i];
				influencePrivate[idxSave] += st[i]*sum_st_inv2;
				st[i] *= sum_st_inv;
				grad_st_private[idxSave] += Ytmp[0]*st[i];
				for (k=1; k < dim; k++) {
					grad_st_private[idxSave + (k*nH)] += Ytmp[k]*st[i];
				}
				grad_st_private[idxSave + dimnH] += st[i];
			}
		}
		#pragma omp critical
		{
			for (i=0; i < nH; i++) {
				influence[i] += influencePrivate[i];
			}
			for (i=0; i < nH*(dim+1); i++) {
				grad_st_tmp[i] += grad_st_private[i];
			}
		}
		free(Ytmp); free(st); free(stInner); free(grad_st_private); free(influencePrivate); free(idxElements);
	} /* end of pragma parallel */

	*TermB = TermBLocal*weight;
	for (i=0; i < nH*(dim+1); i++) {
		gradB[i] -= grad_st_tmp[i]*weight;
	}
	
	free(grad_st_tmp); free(gridLocal); free(aGamma); free(bGamma);
}

