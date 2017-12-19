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
	double *influence = mxGetPr(prhs[7]);
	double *XW = mxGetPr(prhs[8]); /* Weight vector for X */
	int n = (int) mxGetScalar(prhs[9]); /* grid dimensions */
   	unsigned short int *YIdx = (unsigned short int*) mxGetData(prhs[10]);
	int *numEntries = (int*) mxGetData(prhs[11]);
    int *elementList = (int*) mxGetData(prhs[12]);
    int *maxElement = (int*) mxGetData(prhs[13]);
	int *idxEntries = (int*) mxGetData(prhs[14]);
	double *evalGrid = mxGetPr(prhs[15]);

	/* Counting variables */
	int N = mxGetM(prhs[0]); /* number of data points */
	int dim = mxGetN(prhs[0]);
	int M = mxGetNumberOfElements(prhs[14]); /* number of grid points */
	int nH = mxGetNumberOfElements(prhs[3]); /* number of hyperplanes */

	double *grad, *TermA, *TermB;

	plhs[0] = mxCreateDoubleMatrix(nH*(dim+1),1,mxREAL);
    grad = mxGetPr(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
	TermA = mxGetPr(plhs[1]);
	plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
	TermB = mxGetPr(plhs[2]);

	calcGradFastC(numEntries,elementList,maxElement,idxEntries,grad,influence,TermA,TermB,X,XW,grid,YIdx,a,b,gamma,weight,delta,N,M,dim,nH,n,evalGrid);
}
