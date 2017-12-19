#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "headers.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Input variables */
	double *X = mxGetPr(prhs[0]);
	double *grid = mxGetPr(prhs[1]);
	double *a = mxGetPr(prhs[2]);
	double *b = mxGetPr(prhs[3]);
	double gamma = mxGetScalar(prhs[4]);
	double weight = mxGetScalar(prhs[5]);
	double *delta = mxGetPr(prhs[6]);
    int n = (int) mxGetScalar(prhs[7]); /* grid dimensions */
    unsigned short int* YIdx = (unsigned short int*) mxGetData(prhs[8]);
    int *numPointsPerBox = (int*)mxGetData(prhs[9]);
    double *boxEvalPoints = mxGetPr(prhs[10]);
    int MBox = (int) mxGetScalar(prhs[11]);
	
	/* Counting variables */
	int N = mxGetM(prhs[0]); /* number of data points */
	int dim = mxGetN(prhs[0]);
	int M = mxGetN(prhs[8]); /* number of grid points */
	int nH = mxGetNumberOfElements(prhs[3]); /* number of hyperplanes */
    int numBoxes = mxGetNumberOfElements(prhs[9])-1; /* number of active boxes */
	int *elementListSize, *elementList, *numEntries, *maxElement, *idxEntries; 

	plhs[0] = mxCreateNumericMatrix(1, N+M, mxINT32_CLASS, mxREAL);
	numEntries = (int*) mxGetData(plhs[0]);
	plhs[1] = mxCreateNumericMatrix(1, N+M, mxINT32_CLASS, mxREAL);
	maxElement = (int*) mxGetData(plhs[1]);
	plhs[2] = mxCreateNumericMatrix(1, M, mxINT32_CLASS, mxREAL);
	idxEntries = (int*) mxGetData(plhs[2]);

	preCondGradC(&elementList,&elementListSize,numEntries,maxElement,idxEntries,X,grid,YIdx,numPointsPerBox,boxEvalPoints,numBoxes,a,b,gamma,weight,delta,N,M,dim,nH,n,MBox);

 	plhs[3] = mxCreateNumericMatrix(1, *elementListSize, mxINT32_CLASS, mxREAL);
	memcpy((int *)mxGetData(plhs[3]),elementList,*elementListSize*sizeof(int));

	free(elementList); free(elementListSize);
}
