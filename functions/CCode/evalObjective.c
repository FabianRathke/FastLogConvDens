#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <float.h>
#include <limits.h>

extern void makeGridC(double *X, unsigned short int **YIdx, unsigned short int **XToBox, int **numPointsPerBox, double **boxEvalPoints, double *ACVH, double *bCVH, double *box, int *lenY, int *numBoxes, int dim, int lenCVH, int NX, double ratio, int minGridSize, int *NGrid, int *MGrid, double **grid, double *weight);

void evalObjectiveC(double* X, double* XW, double* box, double* a, double* b, double gamma, int dim, int nH, int N, double *ACVH, double *bCVH, int lenCVH, double* evalFunc, double *evalFuncX, double ratio, int minGridSize)
{
	omp_set_num_threads(1);
	int i,j,k,l;
	 // create the integration grid
    int lenY, numBoxes = 0;
    int *numPointsPerBox; unsigned short int *YIdx, *XToBox; double *boxEvalPoints;

    // obtain grid density params
    int NGrid, MGrid;
    double weight = 0;
    double *grid = NULL;

	makeGridC(X,&YIdx,&XToBox,&numPointsPerBox,&boxEvalPoints,ACVH,bCVH,box,&lenY,&numBoxes,dim,lenCVH,N,ratio,minGridSize,&NGrid, &MGrid,&grid,&weight);

    printf("Obtain grid for N = %d and M = %d, %d\n",NGrid,MGrid,N);
    printf("Obtained grid with %d points and %d boxes\n",lenY,numBoxes);

    double *delta = malloc(dim*sizeof(double));
	for (i=0; i < dim; i++) {
        delta[i] = grid[NGrid*MGrid*i+1] - grid[NGrid*MGrid*i];
	}
	
	double *aGamma = malloc(dim*nH*sizeof(double)); 
	double *bGamma = malloc(nH*sizeof(double));
	double factor = 1/gamma;
	double *gridLocal = malloc(dim*sizeof(double));	
	double epsCalcExp = -25; // this is the maximal allowed difference between the maximum and any other hyperplane in log-space; this could be possibly -700, as exp(-700) is roughly the realmin, but we are greedy and drop some hyperplanes
	double TermA = 0, TermAMax = 0, TermB = 0, TermBMax = 0;

	/* initialize some variables */
	for (k=0; k < dim; k++) {
		gridLocal[k] = grid[k*NGrid*MGrid];
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
			//printf("%.4f\n", TermA);
			TermAMax += XW[j]*ftInnerMax*factor;
			evalFuncX[j] = ftInnerMax*factor;
		}
		free(ft); free(ftInner);
	}

	//printf("%d\n", numBoxes);
	/* Calculate gradient for grid points */
	#pragma omp parallel
	{
       	double *Ytmp = calloc(dim,sizeof(double));
        double stInnerMax; double stInnerCorrection = 0;
        double sum_st;
        /*double *st = calloc(nH,sizeof(double)); */
        double *stInner = calloc(nH,sizeof(double));
        /*double *stInnerCheck = calloc(nH,sizeof(double));*/
        double Delta,evalTmp;
        int *idxElements = malloc(nH*sizeof(int));
        int *idxElementsBox = malloc(nH*sizeof(int));
        int numElements, numElementsBox, idxSave, idxMax, sign;
        double *preCalcElems = malloc(nH*dim*MGrid*sizeof(double));
        int YIdxMin, YIdxMax, idxGet; 

		/* calculate gradient for grid points */
		#pragma omp for private(i,k,l) reduction(+:TermB,TermBMax)
   		for (j = 0; j < numBoxes; j++) {
            /* check for active hyperplanes */
            /* eval all hyperplanes for some corner point of the box */
			//printf("Box %d\n", j);
            for (k=0; k < dim; k++) {
                Ytmp[k] = boxEvalPoints[j*3*dim + k];
			//	printf("%.3f, ", Ytmp[k]);
            }
		//	printf("\n");
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
			//printf("Num Elems: %d\n", numElementsBox);
			
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
                    idxSave = l%MGrid;
                    if (k==0) {
                        for (i=0; i < numElementsBox; i++) {
                            preCalcElems[i + idxSave*nH] =  bGamma[idxElementsBox[i]] + aGamma[idxElementsBox[i]]*Ytmp[k];
                        }
                    } else  {
                        for (i=0; i < numElementsBox; i++) {
                            preCalcElems[i + k*MGrid*nH + idxSave*nH] = aGamma[idxElementsBox[i]+k*nH]*Ytmp[k];
                        }
                    }
                }
            }


		   for (l=numPointsPerBox[j]; l < numPointsPerBox[j+1]; l++) {
				stInnerMax = -DBL_MAX;
    			for (k=0; k < dim; k++) {
                    Ytmp[k] = gridLocal[k]+delta[k]*YIdx[k + l*dim];
					//printf("%.4f (%d)", Ytmp[k], YIdx[k + l*dim]);
                }
///				printf("\n");
				
     			idxGet = (YIdx[l*dim]%MGrid)*nH;
                for (i=0; i < numElementsBox; i++) {
                    //stInner[i] = bGamma[idxElementsBox[i]] + aGamma[idxElementsBox[i]]*Ytmp[0]; 
                    stInner[i] = preCalcElems[i + idxGet];
                }
                for (k=1; k < dim; k++) {
                    idxGet = (YIdx[k+l*dim]%MGrid)*nH + k*MGrid*nH;
                    for (i=0; i < numElementsBox; i++) {
                        //stInner[i] += (aGamma[idxElementsBox[i]+k*nH]*Ytmp[k]);
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
				//printf("%d: %.4e\n", l, sum_st-ft_save[l]);
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

	printf("Log-Sum-Exp: %.4f (%.4f, %.5f)\n",TermA + TermB, TermA, TermB);
	printf("Max: %.4f (%.4f, %.5f)\n",TermAMax + TermBMax, TermAMax, TermBMax);
	free(gridLocal); free(aGamma); free(bGamma); free(delta);
	free(numPointsPerBox); free(XToBox); free(boxEvalPoints); free(YIdx); free(grid);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    /* Input variables */
    double *X = mxGetPr(prhs[0]);
    double *a = mxGetPr(prhs[1]);
    double *b = mxGetPr(prhs[2]);
    double gamma = mxGetScalar(prhs[3]);
    double *box  = mxGetData(prhs[4]);
    double *ACVH  = mxGetData(prhs[5]);
    double *bCVH  = mxGetData(prhs[6]);
    double *XW = mxGetPr(prhs[7]); /* Weight vector for X */
    double *evalFunc = mxGetPr(prhs[8]);
    double *evalFuncX = mxGetPr(prhs[9]);
	double ratio = mxGetScalar(prhs[10]);
	int minGridSize = (int) mxGetScalar(prhs[11]);
 

    /* Counting variables */
    int N = mxGetM(prhs[0]); /* number of data points */
    int dim = mxGetN(prhs[0]);
    int nH = mxGetNumberOfElements(prhs[2]); /* number of hyperplanes */
    int lenCVH = mxGetNumberOfElements(prhs[6]);

    evalObjectiveC(X,XW, box, a, b, gamma, dim, nH, N, ACVH, bCVH, lenCVH, evalFunc, evalFuncX, ratio, minGridSize);
}


