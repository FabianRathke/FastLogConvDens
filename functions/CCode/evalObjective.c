#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

extern void evalObjectiveC(double* X, double* XW, double* grid, unsigned short int* YIdx, int *numPointsPerBox, double* boxEvalPoints, int MBox, int numBoxes, double* a, double* b, double gamma, double weight, double* delta, int N, int M, int dim, int nH, int n, double* evalFunc);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	if (nrhs < 10) {
		mexErrMsgTxt("Not enough input variable");
	}
			
	/* Input variables */
	double *X = mxGetPr(prhs[0]);
	double *grid = mxGetPr(prhs[1]);
	double *a = mxGetPr(prhs[2]);
	double *b = mxGetPr(prhs[3]);
	double gamma = mxGetScalar(prhs[4]);
	double weight = mxGetScalar(prhs[5]);
	double *delta = mxGetPr(prhs[6]);
	double *XW = mxGetPr(prhs[7]); /* Weight vector for X */
	int n = (int) mxGetScalar(prhs[8]); /* grid dimensions */
	unsigned short int* YIdx = (unsigned short int*) mxGetData(prhs[9]);
    int *numPointsPerBox = (int*)mxGetData(prhs[10]);
    double *boxEvalPoints = mxGetPr(prhs[11]);
    int MBox = (int) mxGetScalar(prhs[12]);
	double *evalFunc = mxGetPr(prhs[13]);

	/* Counting variables */
	int N = mxGetM(prhs[0]); /* number of data points */
	int dim = mxGetN(prhs[0]);
	int M = mxGetN(prhs[9]); /* number of grid points */
	int nH = mxGetNumberOfElements(prhs[3]); /* number of hyperplanes */
   	int numBoxes = mxGetNumberOfElements(prhs[10])-1; /* number of active boxes */

	evalObjectiveC(X,XW,grid,YIdx,numPointsPerBox,boxEvalPoints,MBox,numBoxes,a,b,gamma,weight,delta,N,M,dim,nH,n,evalFunc);
}
