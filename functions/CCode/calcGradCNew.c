#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <float.h>
#include <limits.h>

void calcGradC(double* gradA, double* gradB, double* influence, double* TermA, double* TermB, double* X, double* XW, double* grid, unsigned short int* YIdx, int *numPointsPerBox, double* boxEvalPoints, unsigned short int *XToBox, int numBoxes, double* a, double* b, double gamma, double weight, double* delta, int N, int M, int dim, int nH, int n, int MBox, double* evalFunc)
{
	double *grad_st_tmp = calloc(nH*(dim+1),sizeof(double));
	double *aGamma = malloc(dim*nH*sizeof(double)); 
	double *bGamma = malloc(nH*sizeof(double));
	int dimnH = dim*nH;
	int i,j,k,l;
	double factor = 1/gamma;
	double *gridLocal = malloc(dim*sizeof(double));	
	double epsCalcExp = -25; /* this is the maximal difference between the maximum and any other hyperplane in log-space; increase this value for more accuracy */
	double TermALocal, TermBLocal;
	int XCounterGlobal = 0;	

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
		gradA[i] = 0;
		gradB[i] = 0;
	}

	printf("new\n");

	/* Calculate gradient for grid points */ 
	TermBLocal = 0; *TermB = 0;
	TermALocal = 0;	*TermA = 0;
	#pragma omp parallel num_threads(NUMCORES)
	{
		double *Ytmp = calloc(dim,sizeof(double));
		double stInnerMax; double stInnerCorrection = 0;
		double sum_st, sum_st_inv, tmpVal, sum_st_inv2;
		/*double *st = calloc(nH,sizeof(double)); */
		double *stInner = calloc(nH,sizeof(double));
		/*double *stInnerCheck = calloc(nH,sizeof(double));*/
		double *aGammaTmp = malloc(dim*nH*sizeof(double));
		double *bGammaTmp = malloc(nH*sizeof(double));
		double *grad_st_private = calloc(nH*(dim+1),sizeof(double));
		double *influencePrivate = calloc(nH,sizeof(double));
		double evalTmp;
		double *Delta = malloc(dim*sizeof(double));
		double *aMax = malloc(dim*sizeof(double));
		int *idxElements = malloc(nH*sizeof(int));
		int *idxElementsBox = malloc(nH*sizeof(int));
		int numElements, numElementsBox, idxSave, idxMax, sign;
		double *preCalcElems = malloc(nH*dim*MBox*sizeof(double));
		int *idxGet = malloc(dim*sizeof(int));
		int YIdxMin, YIdxMax,totalHyperplanes = 0;

		double stTmp;
		int XCounter=0; /* counts how much X elements were allready processed */
        double *grad_ft_private = calloc(nH*(dim+1),sizeof(double));

		/* calculate gradient for grid points */
		#pragma omp for schedule(dynamic,1) private(i,k,l) reduction(+:TermBLocal,TermALocal)
		for (j = 0; j < numBoxes; j++) {
			/* check for active hyperplanes */
			/* if there is more than one active grid point in that box (active means inside the convex hull of X) check for active hyperplanes */
			if (numPointsPerBox[j+1] - numPointsPerBox[j] > 1) { 
				/* eval all hyperplanes for some corner point of the box */
				for (k=0; k < dim; k++) {
					Ytmp[k] = boxEvalPoints[j*3*dim + k];
				}
				stInnerMax = -DBL_MAX;
				for (i=0; i < nH; i++) {
					stInner[i] = bGamma[i] + aGamma[i]*Ytmp[0];
		
					for (k=1; k < dim; k++) {
						stInner[i] += aGamma[i+k*nH]*Ytmp[k];
					}
					if (stInner[i] > stInnerMax) {
						stInnerMax = stInner[i];
						idxMax = i;
					}
				}				

				numElementsBox = 0;
				for (k=0; k < dim; k++) {
					Delta[k] = boxEvalPoints[j*3*dim + 1*dim + k] * boxEvalPoints[j*3*dim + 2*dim + k];
					aMax[k] = aGamma[idxMax + k*nH];
				}
				/* eval all hyperplanes for the point opposite to the first one */
				for (i=0; i < nH; i++) {
					tmpVal = 0;
					evalTmp = (aGamma[i]-aMax[0])*Delta[0];
					if (evalTmp > 0) {
						tmpVal += evalTmp;
					}

					for (k=1; k < dim; k++) {
						evalTmp = (aGamma[i+k*nH]-aMax[k])*Delta[k];
						if (evalTmp > 0) {
							tmpVal += evalTmp;
						}
					}
					stInner[i] += tmpVal;
					if (stInner[i]  > stInnerMax - 25) {
						idxElementsBox[numElementsBox++] = i;
					}
				}
			/* otherwise, assume all hyperplanes active for the single grid point in that box*/
			} else {
				/* check which hyperplanes to keep for that box */
				numElementsBox = nH;
				for (i=0; i < nH; i++) {
					idxElementsBox[i] = i;
				}
			}

			for (i =0; i < numElementsBox; i++) {
				bGammaTmp[i] = bGamma[idxElementsBox[i]];
				for (k=0; k < dim; k++) {
					aGammaTmp[i + k*nH] = aGamma[idxElementsBox[i] + k*nH];
				}

			}
			totalHyperplanes += numElementsBox;
			
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
					} else 	{
						for (i=0; i < numElementsBox; i++) {
							preCalcElems[i + k*MBox*nH + idxSave*nH] = aGamma[idxElementsBox[i]+k*nH]*Ytmp[k];
						}
					}
				}
			}

			/* Move XCounter to the current box */
			while (XToBox[XCounter] < j) {
				XCounter++;
			}
			
			/* iterate over all samples in that box */
			while (XToBox[XCounter] == j) {
			//while (0) {			
				stInnerMax = -DBL_MAX; numElements = 0;
				for (i=0; i < numElementsBox; i++) {
					stTmp = bGammaTmp[i] + aGammaTmp[i]*X[XCounter];
					for (k=1; k < dim; k++) {
						stTmp += aGammaTmp[i+k*nH]*X[XCounter + (k*N)];
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
				// calculate st only for those entries that will be non-zero */
				for (i=0; i < numElements; i++) {
					stInner[i] = exp(stInner[i]-stInnerMax);
					sum_st += stInner[i];
				}

				TermALocal += XW[XCounter]*(stInnerMax + log(sum_st))*factor;
				sum_st_inv = 1/sum_st*XW[XCounter];

				/* update the gradient */
				for (i=0; i < numElements; i++) {
					idxSave = idxElementsBox[idxElements[i]];
					stTmp = stInner[i]*sum_st_inv;
					for (k=0; k < dim; k++) {
						grad_ft_private[idxSave + (k*nH)] += stTmp*X[XCounter+(k*N)];
					}
					grad_ft_private[idxSave + (dim*nH)] += stTmp;
				}
				XCounter++;
			}
			

			/* iterate over all points in that box */
			for (l=numPointsPerBox[j]; l < numPointsPerBox[j+1]; l++) {
//			for (l=0; l < 0; l++) {			
/*				printf("%d ",l); */
				stInnerMax = -DBL_MAX; numElements = 0;
				for (k=0; k < dim; k++) {
					Ytmp[k] = gridLocal[k]+delta[k]*YIdx[k + l*dim];
				}
				for (k = 0; k < dim; k++) {
					idxGet[k] = (YIdx[k+l*dim]%MBox)*nH + k*MBox*nH;
				}

				for (i=0; i < numElementsBox; i++) {
//					stTmp = bGammaTmp[i] + aGammaTmp[i]*Ytmp[0];
					stTmp = preCalcElems[i + idxGet[0]];
					for (k=1; k < dim; k++) {
//						stTmp += (aGammaTmp[i+k*nH]*Ytmp[k]);
						stTmp += preCalcElems[i + idxGet[k]];
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
				for (i=0; i < numElements; i++) {
					stInner[i] = exp(stInner[i]-stInnerMax);
					sum_st += stInner[i];
				}

				stInnerCorrection = exp(-stInnerMax*factor);
				tmpVal = pow(sum_st,-factor)*stInnerCorrection;
				
				TermBLocal += tmpVal; evalFunc[l] = tmpVal;
				sum_st_inv2 = 1/sum_st;
				sum_st_inv = tmpVal*sum_st_inv2;
				
				for (i=0; i < numElements; i++) {
					idxSave = idxElementsBox[idxElements[i]];
					influencePrivate[idxSave] += stInner[i]*sum_st_inv2;
					stInner[i] *= sum_st_inv;
					grad_st_private[idxSave] += Ytmp[0]*stInner[i];
					for (k=1; k < dim; k++) {
						grad_st_private[idxSave + (k*nH)] += Ytmp[k]*stInner[i];
					}
					grad_st_private[idxSave + dimnH] += stInner[i];
				}
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
	      	for (i=0; i < nH*(dim+1); i++) {
                gradA[i] += grad_ft_private[i];
            }

		}
		free(Ytmp); free(stInner); free(grad_st_private); free(grad_ft_private); free(influencePrivate); free(idxElements); free(idxElementsBox); free(preCalcElems); /* free(st); free(stCheck) */
	} /* end of pragma parallel */
	*TermB = TermBLocal*weight;
	for (i=0; i < nH*(dim+1); i++) {
		gradB[i] -= grad_st_tmp[i]*weight;
	}

	
	/* move X pointer to elements that are not contained in any box */
	while(XToBox[XCounterGlobal] != 65535) {
		XCounterGlobal++;
	}
	
	/* calculate gradient for samples X */ 
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
        #pragma omp for private(i,k) schedule(dynamic) reduction(+:TermALocal)
		for (j=XCounterGlobal; j < N; j++) {
//		for (j=0; j < 0; j++) {		
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
	free(grad_st_tmp); free(gridLocal); free(aGamma); free(bGamma);
}
