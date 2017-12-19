#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

extern void calcGradFastC(int* numEntries, int* elementList, int* maxElement, int* idxEntries, float* grad, double* influence, float* TermA, float* TermB, float* X, float* XW, float* grid, unsigned short int* YIdx, float* a, float* b, float gamma, float weight, float* delta, int N, int M, int dim, int nH, int n);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Input variables */
	float *X = (float*) mxGetData(prhs[0]);
	float *grid = (float*) mxGetData(prhs[1]);
	float *a = (float*) mxGetData(prhs[2]);
	float *b = (float*) mxGetData(prhs[3]);
	float gamma = mxGetScalar(prhs[4]);
	float weight = mxGetScalar(prhs[5]);
	float *delta = (float*) mxGetData(prhs[6]);
	double *influence = mxGetPr(prhs[7]);
	float *XW = (float*) mxGetData(prhs[8]); /* Weight vector for X */
	int n = (int) mxGetScalar(prhs[9]); /* grid dimensions */
   	unsigned short int *YIdx = (unsigned short int*) mxGetData(prhs[10]);
	int *numEntries = (int*) mxGetData(prhs[11]);
    int *elementList = (int*) mxGetData(prhs[12]);
    int *maxElement = (int*) mxGetData(prhs[13]);
	int *idxEntries = (int*) mxGetData(prhs[14]);
//	float *evalGrid = (float*) mxGetData(prhs[15]);

	/* Counting variables */
	int N = mxGetM(prhs[0]); /* number of data points */
	int dim = mxGetN(prhs[0]);
	int M = mxGetNumberOfElements(prhs[14]); /* number of grid points */
	int nH = mxGetNumberOfElements(prhs[3]); /* number of hyperplanes */

	float *grad, *TermA, *TermB;

	plhs[0] = mxCreateNumericMatrix(nH*(dim+1),1,mxSINGLE_CLASS,mxREAL);
    grad = (float*) mxGetData(plhs[0]);
	plhs[1] = mxCreateNumericMatrix(1,1,mxSINGLE_CLASS,mxREAL);
	TermA = (float*) mxGetData(plhs[1]);
	plhs[2] = mxCreateNumericMatrix(1,1,mxSINGLE_CLASS,mxREAL);
	TermB = (float*) mxGetData(plhs[2]);

	calcGradFastC(numEntries,elementList,maxElement,idxEntries,grad,influence,TermA,TermB,X,XW,grid,YIdx,a,b,gamma,weight,delta,N,M,dim,nH,n);
}
