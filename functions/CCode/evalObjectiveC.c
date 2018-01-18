#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <float.h>
#include <limits.h>

void evalObjectiveC(double* X, double* XW, double* grid, unsigned short int* YIdx, int *numPointsPerBox, double* boxEvalPoints, int MBox, int numBoxes, double* a, double* b, double gamma, double weight, double* delta, int N, int M, int dim, int nH, int n, double* evalFunc)
{
	double *aGamma = malloc(dim*nH*sizeof(double)); 
	double *bGamma = malloc(nH*sizeof(double));
	int i,j,k,l;
	double factor = 1/gamma;
	double *gridLocal = malloc(dim*sizeof(double));	
	double epsCalcExp = -25; // this is the maximal allowed difference between the maximum and any other hyperplane in log-space; this could be possibly -700, as exp(-700) is roughly the realmin, but we are greedy and drop some hyperplanes
	double TermA = 0, TermAMax = 0, TermB = 0, TermBMax = 0;
	
	/* initialize some variables */
	for (k=0; k < dim; k++) {
		gridLocal[k] = grid[k];
	}

	for (i=0; i < nH; i++) {
		for (j=0; j < dim; j++) {
			aGamma[i+(j*nH)] = gamma*a[i+(j*nH)];
		}
		bGamma[i] = gamma*b[i];
	}

	/* calculate gradient for samples */
	#pragma omp parallel 
    {
   	  	double ftInnerMax;
        double sum_ft;
        double *ft = calloc(nH,sizeof(double));
        double *ftInner = calloc(nH,sizeof(double));
		int numElements, idxElements[nH];
        /* calculate gradient for samples */
        #pragma omp for private(i,k) reduction(+:TermA,TermAMax)
		for (j=0; j < N; j++) {
			ftInnerMax = -DBL_MAX;
			for (i=0; i < nH; i++) {
				ftInner[i] = bGamma[i] + aGamma[i]*X[j];
			}
			for (k=1; k < dim; k++) {
				for (i=0; i < nH; i++) {
					ftInner[i] += aGamma[i+k*nH]*X[j + (k*N)];
				}
			}

			// find maximum element
			for (i=0; i < nH; i++) {
				if (ftInner[i] > ftInnerMax) {
					ftInnerMax = ftInner[i];
				}
			}

			sum_ft = 0; numElements = 0;
			// calculate ft only for those entries that are will be non-zero
			for (i=0; i < nH; i++) {
				if (ftInner[i] - ftInnerMax > epsCalcExp) {
					ft[numElements] = exp(ftInner[i]-ftInnerMax);
                    idxElements[numElements] = i;
					sum_ft += ft[numElements++];
				}
			}

			TermA += XW[j]*(ftInnerMax + log(sum_ft))*factor;
			TermAMax += XW[j]*ftInnerMax*factor;
		}
		free(ft); free(ftInner);
	}

	/* Calculate gradient for grid points */
	#pragma omp parallel
	{
       	double *Ytmp = calloc(dim,sizeof(double));
        double stInnerMax; double stInnerCorrection = 0;
        double sum_st, sum_st_inv, tmpVal;
        /*double *st = calloc(nH,sizeof(double)); */
        double *stInner = calloc(nH,sizeof(double));
        /*double *stInnerCheck = calloc(nH,sizeof(double));*/
        double Delta,evalTmp;
        int *idxElements = malloc(nH*sizeof(int));
        int *idxElementsBox = malloc(nH*sizeof(int));
        int numElements, numElementsBox, idxSave, idxMax, sign;
        double *preCalcElems = malloc(nH*dim*MBox*sizeof(double));
        int YIdxMin, YIdxMax, idxGet, totalHyperplanes = 0;

		/* calculate gradient for grid points */
		#pragma omp for private(i,k,l) reduction(+:TermB,TermBMax)
   		for (j = 0; j < numBoxes; j++) {
            /* check for active hyperplanes */
            /* eval all hyperplanes for some corner point of the box */
            for (k=0; k < dim; k++) {
                Ytmp[k] = boxEvalPoints[j*3*dim + k];
            }
            stInnerMax = -DBL_MAX;
            for (i=0; i < nH; i++) {
                stInner[i] = bGamma[i] + aGamma[i]*Ytmp[0];
            }
            for (k=1; k < dim; k++) {
                for (i=0; i < nH; i++) {
                    stInner[i] += aGamma[i+k*nH]*Ytmp[k];
                }
            }

            /* find maximum element for current grid point */
            for (i=0; i < nH; i++) {
                if (stInner[i] > stInnerMax) {
                    stInnerMax = stInner[i];
                    idxMax = i;
                }
            }
			
			/* if more than one grid point in that box */
			if (numPointsPerBox[j+1] - numPointsPerBox[j] > 1) {
				/* eval all hyperplanes for the point opposite to the first one */
				sign = boxEvalPoints[j*3*dim + 2*dim];
				Delta = boxEvalPoints[j*3*dim + 1*dim];
				for (i=0; i < nH; i++) {
					evalTmp = (aGamma[i]-aGamma[idxMax])*sign;
					/*stInnerCheck[i] = 0;*/
					if (evalTmp > 0) {
						stInner[i] += evalTmp*Delta;
					}
				}

				for (k=1; k < dim; k++) {
					sign = boxEvalPoints[j*3*dim + 2*dim + k];
					Delta = boxEvalPoints[j*3*dim + 1*dim + k];
					for (i=0; i < nH; i++) {
						evalTmp = (aGamma[i+k*nH]-aGamma[idxMax + k*nH])*sign;
						if (evalTmp > 0) {
							stInner[i] += evalTmp*Delta;
						}
					}
				}
			}

            /* check which hyperplanes to keep for that box */
            numElementsBox = 0;
            for (i=0; i < nH; i++) {
                if (stInner[i] > stInnerMax + epsCalcExp) {
                    idxElementsBox[numElementsBox++] = i;
                }
            }
			
            /* precalc elements for that box */
            for (k=0; k < dim; k++) {
                YIdxMin = INT_MAX; YIdxMax = INT_MIN;
                for (l=numPointsPerBox[j]; l < numPointsPerBox[j+1]; l++) {
                    if (YIdxMin > YIdx[k + l*dim]) {
                        YIdxMin = YIdx[k + l*dim];
                    }
                    if (YIdxMax < YIdx[k + l*dim]) {
                        YIdxMax = YIdx[k + l*dim];
                    }
                }

                for (l=YIdxMin; l <= YIdxMax; l++) {
                    Ytmp[k] = gridLocal[k]+delta[k]*l;
                    idxSave = l%MBox;
                    if (k==0) {
                        for (i=0; i < numElementsBox; i++) {
                            preCalcElems[i + idxSave*nH] =  bGamma[idxElementsBox[i]] + aGamma[idxElementsBox[i]]*Ytmp[k];
                        }
                    } else  {
                        for (i=0; i < numElementsBox; i++) {
                            preCalcElems[i + k*MBox*nH + idxSave*nH] = aGamma[idxElementsBox[i]+k*nH]*Ytmp[k];
                        }
                    }
                }
            }


		   for (l=numPointsPerBox[j]; l < numPointsPerBox[j+1]; l++) {
				stInnerMax = -DBL_MAX;
    			for (k=0; k < dim; k++) {
                    Ytmp[k] = gridLocal[k]+delta[k]*YIdx[k + l*dim];
                }
     			idxGet = (YIdx[l*dim]%MBox)*nH;
                for (i=0; i < numElementsBox; i++) {
                    /* stInner[i] = bGamma[idxElementsBox[i]] + aGamma[idxElementsBox[i]]*Ytmp[0];   */
                    stInner[i] = preCalcElems[i + idxGet];
                }
                for (k=1; k < dim; k++) {
                    idxGet = (YIdx[k+l*dim]%MBox)*nH + k*MBox*nH;
                    for (i=0; i < numElementsBox; i++) {
                        /* stInner[i] += (aGamma[idxElementsBox[i]+k*nH]*Ytmp[k]);  */
                        stInner[i] += preCalcElems[i + idxGet];
                    }
                }

                /* find maximum element for current grid point */
                for (i=0; i < numElementsBox; i++) {
                    if (stInner[i] > stInnerMax) {
                        stInnerMax = stInner[i];
                    }
                }
				
				// calculate st only for those entries that wont be zero
  				sum_st = 0; numElements = 0;
                for (i=0; i < numElementsBox; i++) {
                    if (stInner[i] - stInnerMax > epsCalcExp) {
                        stInner[numElements] = exp(stInner[i]-stInnerMax);
                        idxElements[numElements] = i;
                        sum_st += stInner[numElements++];
                    }
                }

				stInnerCorrection = exp(-stInnerMax*factor);
				TermB += pow(sum_st,-factor)*stInnerCorrection;
				/* evaluate density for max function --> not the gamma approximation */
				evalFunc[l] = stInnerCorrection;
				TermBMax += stInnerCorrection;
			}	
		}
		free(Ytmp); free(stInner); free(idxElements); free(idxElementsBox); free(preCalcElems);
	} // end of pragma parallel
	TermB = TermB*weight; TermBMax = TermBMax*weight;

	printf("Log-Sum-Exp: %.4f (%.4f, %.5f)\n",TermA/N + TermB, TermA/N, TermB);
	printf("Max: %.4f (%.4f, %.5f)\n",TermAMax/N + TermBMax, TermAMax/N, TermBMax);
	free(gridLocal); free(aGamma); free(bGamma);
}

