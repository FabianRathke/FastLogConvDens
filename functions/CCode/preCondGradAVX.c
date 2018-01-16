#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

extern void preCondGradAVXC(int** elementList, int** elementListSize, int* numEntries, int* maxElement, int* idxEntries, float* X, float* grid, unsigned short int* YIdx, int *numPointsPerBox, float* boxEvalPoints, int numBoxes, double* a, double* aTrans, double* b, float gamma, float weight, float* delta, int N, int M, int dim, int nH);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Input variables */
	float *X = (float*) mxGetData(prhs[0]);
	float *grid = (float*) mxGetData(prhs[1]);
	double *a = (double*) mxGetData(prhs[2]);
	double *aTrans = (double*) mxGetData(prhs[10]);
	double *b = (double*) mxGetData(prhs[3]);
	float gamma = (float) mxGetScalar(prhs[4]);
	float weight = (float) mxGetScalar(prhs[5]);
	float *delta = (float*) mxGetData(prhs[6]);
    unsigned short int* YIdx = (unsigned short int*) mxGetData(prhs[7]);
    int *numPointsPerBox = (int*)mxGetData(prhs[8]);
    float *boxEvalPoints = (float*) mxGetData(prhs[9]);
	
	/* Counting variables */
	int N = mxGetM(prhs[0]); /* number of data points */
	int dim = mxGetN(prhs[0]);
	int M = mxGetN(prhs[7]); /* number of grid points */
	int nH = mxGetNumberOfElements(prhs[3]); /* number of hyperplanes */
    int numBoxes = mxGetNumberOfElements(prhs[8])-1; /* number of active boxes */
	int *elementListSize, *elementList, *numEntries, *maxElement, *idxEntries; 

	plhs[0] = mxCreateNumericMatrix(1, N+M, mxINT32_CLASS, mxREAL);
	numEntries = (int*) mxGetData(plhs[0]);
	plhs[1] = mxCreateNumericMatrix(1, N+M, mxINT32_CLASS, mxREAL);
	maxElement = (int*) mxGetData(plhs[1]);
	plhs[2] = mxCreateNumericMatrix(1, M, mxINT32_CLASS, mxREAL);
	idxEntries = (int*) mxGetData(plhs[2]);

	preCondGradAVXC(&elementList,&elementListSize,numEntries,maxElement,idxEntries,X,grid,YIdx,numPointsPerBox,boxEvalPoints,numBoxes,a,aTrans,b,gamma,weight,delta,N,M,dim,nH);
 	plhs[3] = mxCreateNumericMatrix(1, *elementListSize, mxINT32_CLASS, mxREAL);
	memcpy((int *)mxGetData(plhs[3]),elementList,*elementListSize*sizeof(int));

	free(elementList); free(elementListSize);
}
