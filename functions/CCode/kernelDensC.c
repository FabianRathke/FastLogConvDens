#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Input variables */
	double *X = mxGetPr(prhs[0]);
	double *sampleWeights = mxGetPr(prhs[1]);
	double *h = mxGetPr(prhs[2]);
	double *sparseGrid = mxGetPr(prhs[3]);
	double *sparseDelta = mxGetPr(prhs[4]);
	unsigned int *idxStart = (unsigned int*) mxGetData(prhs[5]);
	unsigned int *numPoints = (unsigned int*) mxGetData(prhs[6]);

	/* Counting variables */
	int N = mxGetN(prhs[0]); /* number of basis points */
	int d = mxGetM(prhs[0]);
	int numBoxes = mxGetNumberOfElements(prhs[5]); /* number of active boxes */

	double *kernelDens = NULL;

	int i;
	const double pi = 3.141592653589793115997963468544185161590576171875;
	double *weights = malloc(d*sizeof(double));
	double *hInv = malloc(d*sizeof(double));
	double normalization = 1;

	plhs[0] = mxCreateDoubleMatrix(N,1,mxREAL);
    kernelDens = mxGetPr(plhs[0]);

	/* calculate normalization constants */
	for (i=0; i < d; i++) {
		weights[i] = 1/(h[i]*sqrt(2*pi));
		normalization *= weights[i]; 
		hInv[i] = -0.5/(h[i]*h[i]);
	}

	/* use this variant only if there are a lot of data points */
	if (N > numBoxes*10) {
		#pragma omp parallel num_threads(NUMCORES)
		{
			double innerSum, yTmp, x, y;
			int *idxElements = malloc(numBoxes*sizeof(int));
			int *idxElementsFine = malloc(numBoxes*sizeof(int));
			int numElements, numElementsFine, assign;
			double *XCurr = malloc(d*sizeof(double));
			int j,k,l,m;
			int idxX,idxY;

			#pragma omp for schedule(dynamic,1)
			for (i=0; i < numBoxes; i++) {
				numElements = 0;
				for (j=0; j < numBoxes; j++) {
					assign = 1;
					/* check if both boxes should be checked */
					for (k=0; k < d; k++) {
						x = sparseGrid[i*d+k];
						y = sparseGrid[j*d+k];
						if(x > y) {
							y += sparseDelta[k];
						} else if (x < y) {
							x += sparseDelta[k];
						} 
						
						/* check if closest points of both boxes are more than 3 standard deviations away from each other */
						if (fabs(x-y) > h[k]*3) {
							assign = 0;
							break;
						}
					}
					/* assign box to list of active boxes */	
					if (assign==1) {
						idxElements[numElements++] = j;
					}
				}

				/* iterate over all data points in box i */
				idxX = idxStart[i];
				for (j=idxX; j < idxX+numPoints[i]; j++) {
					/* extract current X */
					for (k=0; k < d; k++) {
						XCurr[k] = X[j*d + k];
					}

					/* check all boxes for this point (make finer selection of required boxes) */
					numElementsFine = 0;
					for (l=0; l < numElements; l++) {
						assign = 1;
						/* always take the same box */
						if (idxElements[l] != i) {
							for (k=0; k < d; k++) {
								x = fabs(sparseGrid[idxElements[l]*d+k]-XCurr[k]);
								y = fabs(sparseGrid[idxElements[l]*d+k]+sparseDelta[k]-XCurr[k]);
								/* check if closest points of both boxes are more than 2 standard deviations away from each other */
								if (x > h[k]*3 && y > h[k]*3) {
									assign = 0;
									break;
								}
							}
						}
						if (assign==1) {
							idxElementsFine[numElementsFine++] = idxElements[l];
						}
					}

					/* for this finer selection evaluate all X's in these boxes */
					yTmp = 0;
					/* fine selection of boxes */
					for (l=0; l < numElementsFine; l++) {
						/* all points in that box */
						for (m = idxStart[idxElementsFine[l]]; m < idxStart[idxElementsFine[l]]+numPoints[idxElementsFine[l]]; m++) {
							idxY = m*d;
							/* for each point we need to evaluate 1-D Gauss Kernels and multiply them */
							innerSum = 0;
							for (k = 0; k < d; k++) {
								innerSum += (XCurr[k]-X[idxY+k])*(XCurr[k]-X[idxY+k])*hInv[k];
							}
							yTmp += sampleWeights[m]*exp(innerSum)*normalization;
						}
					}
					kernelDens[j] = yTmp;
				}
			}
			free(idxElements); free(idxElementsFine); free(XCurr);
		}	
	} else {
		#pragma omp parallel num_threads(NUMCORES)
		{   
			int idxX,idxY,j,k;
			double innerSum,yTmp;
			/* for each point in Y calculate kernel density */
			#pragma omp for
			for (i=0; i < N; i++) {
				idxX = i*d;
				/* evalute the Gauss kernel for each point in X and multiply with sampleWeight */
				yTmp = 0;
				for (j = 0; j < N; j++) {
					idxY = j*d;
					/* for each point we need to evaluate 1-D Gauss Kernels and multiply them */
					innerSum = 0;
					for (k = 0; k < d; k++) {
						innerSum += (X[idxX+k]-X[idxY+k])*(X[idxX+k]-X[idxY+k])*hInv[k];
					}
					yTmp += sampleWeights[j]*exp(innerSum)*normalization;
				}
				kernelDens[i] = yTmp;
			}
		}
	}

	free(weights); free(hInv);
}
