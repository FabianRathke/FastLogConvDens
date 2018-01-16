#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

extern void calcGradFastC(int* numEntries, int* elementList, int* maxElement, int* idxEntries, double* grad, double* influence, double* TermA, double* TermB, float* X, float* XW, float* grid, unsigned short int* YIdx, double* a, double* b, float gamma, float weight, float* delta, int N, int M, int dim, int nH);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Input variables */
	float *X = (float*) mxGetData(prhs[0]);
	float *grid = (float*) mxGetData(prhs[1]);
	double *a = (double*) mxGetData(prhs[2]);
	double *b = (double*) mxGetData(prhs[3]);
	float gamma = mxGetScalar(prhs[4]);
	float weight = mxGetScalar(prhs[5]);
	float *delta = (float*) mxGetData(prhs[6]);
	double *influence = mxGetPr(prhs[7]);
	float *XW = (float*) mxGetData(prhs[8]); /* Weight vector for X */
   	unsigned short int *YIdx = (unsigned short int*) mxGetData(prhs[9]);
	int *numEntries = (int*) mxGetData(prhs[10]);
    int *elementList = (int*) mxGetData(prhs[11]);
    int *maxElement = (int*) mxGetData(prhs[12]);
	int *idxEntries = (int*) mxGetData(prhs[13]);
//	float *evalGrid = (float*) mxGetData(prhs[15]);

	/* Counting variables */
	int N = mxGetM(prhs[0]); /* number of data points */
	int dim = mxGetN(prhs[0]);
	int M = mxGetNumberOfElements(prhs[13]); /* number of grid points */
	int nH = mxGetNumberOfElements(prhs[3]); /* number of hyperplanes */

	double *grad, *TermA, *TermB;

	plhs[0] = mxCreateNumericMatrix(nH*(dim+1),1,mxDOUBLE_CLASS,mxREAL);
    grad = (double*) mxGetData(plhs[0]);
	plhs[1] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
	TermA = (double*) mxGetData(plhs[1]);
	plhs[2] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS,mxREAL);
	TermB = (double*) mxGetData(plhs[2]);

	calcGradFastC(numEntries,elementList,maxElement,idxEntries,grad,influence,TermA,TermB,X,XW,grid,YIdx,a,b,gamma,weight,delta,N,M,dim,nH);
}
