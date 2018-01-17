#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

extern void calcGradAVXC(double* gradA, double* gradB, double* influence, double* TermA, double* TermB, float* X, float* XW, float* grid, unsigned short int* YIdx, int *numPointsPerBox, float* boxEvalPoints, unsigned short int *XToBox, int numBoxes, double* a, double* b, float gamma, float weight, float* delta, int N, int M, int dim, int nH, int MBox);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Matlab input variables */
	float *X = (float*) mxGetData(prhs[0]);
	float *grid = (float*) mxGetData(prhs[1]);
	double *a = (double*) mxGetData(prhs[2]); // transposed version
	double *b = (double*) mxGetData(prhs[3]);
	float gamma = mxGetScalar(prhs[4]);
	float weight = mxGetScalar(prhs[5]);
	float *delta = (float*) mxGetData(prhs[6]);
	double *influence =  mxGetPr(prhs[7]);
	float *XW = (float*) mxGetData(prhs[8]); /* Weight vector for X */
	int n = (int) mxGetScalar(prhs[9]); /* grid dimensions */
	unsigned short int* YIdx = (unsigned short int*) mxGetData(prhs[10]);
	int *numPointsPerBox = (int*)mxGetData(prhs[11]);
	float *boxEvalPoints = (float*) mxGetData(prhs[12]);
	unsigned short int *XToBox = (unsigned short int*)mxGetData(prhs[13]);
	int MBox = (int) mxGetScalar(prhs[14]);
	float *evalFunc = (float*) mxGetData(prhs[15]);

	/* Counting variables */
	int N = mxGetM(prhs[0]); /* number of data points */
	int dim = mxGetN(prhs[0]);
	int M = mxGetN(prhs[10]); /* number of grid points */
	int nH = mxGetNumberOfElements(prhs[3]); /* number of hyperplanes */
	int numBoxes = mxGetNumberOfElements(prhs[11])-1; /* number of active boxes */
	double *gradA, *gradB, *TermA, *TermB;

	plhs[0] = mxCreateNumericMatrix(nH*(dim+1),1,mxDOUBLE_CLASS,mxREAL);
    gradA = (double*) mxGetData(plhs[0]);
	plhs[1] = mxCreateNumericMatrix(nH*(dim+1),1,mxDOUBLE_CLASS,mxREAL);
    gradB = (double*) mxGetData(plhs[1]);
	plhs[2] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
	TermA = (double*) mxGetData(plhs[2]);
	plhs[3] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
	TermB = (double*) mxGetData(plhs[3]);

	calcGradAVXC(gradA,gradB,influence,TermA,TermB,X,XW,grid,YIdx,numPointsPerBox,boxEvalPoints,XToBox,numBoxes,a,b,gamma,weight,delta,N,M,dim,nH,MBox);
}
