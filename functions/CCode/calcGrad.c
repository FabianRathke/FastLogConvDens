#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "headers.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Matlab input variables */
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
	unsigned short int* YIdx = (unsigned short int*) mxGetData(prhs[10]);
	int *numPointsPerBox = (int*)mxGetData(prhs[11]);
	double *boxEvalPoints = mxGetPr(prhs[12]);
	unsigned short int *XToBox = (unsigned short int*)mxGetData(prhs[13]);
	int MBox = (int) mxGetScalar(prhs[14]);
	double *evalFunc = mxGetPr(prhs[15]);

	/* Counting variables */
	int N = mxGetM(prhs[0]); /* number of data points */
	int dim = mxGetN(prhs[0]);
	int M = mxGetN(prhs[10]); /* number of grid points */
	int nH = mxGetNumberOfElements(prhs[3]); /* number of hyperplanes */
	int numBoxes;	
	double *gradA, *gradB, *TermA, *TermB;

	plhs[0] = mxCreateDoubleMatrix(nH*(dim+1),1,mxREAL);
    gradA = mxGetPr(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(nH*(dim+1),1,mxREAL);
    gradB = mxGetPr(plhs[1]);
	plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
	TermA = mxGetPr(plhs[2]);
	plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL);
	TermB = mxGetPr(plhs[3]);

	numBoxes = mxGetNumberOfElements(prhs[11])-1; /* number of active boxes */
	calcGradC(gradA,gradB,influence,TermA,TermB,X,XW,grid,YIdx,numPointsPerBox,boxEvalPoints,XToBox,numBoxes,a,b,gamma,weight,delta,N,M,dim,nH,MBox,evalFunc); 
}
