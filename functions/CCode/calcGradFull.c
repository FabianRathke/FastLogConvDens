#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "headers.h"

extern void calcGradFullC(double* gradA, double* gradB, double* influence, double* TermA, double* TermB, double* X, double* XW, double* grid, unsigned short int* YIdx, double* a, double* b, double gamma, double weight, double* delta, int N, int M, int dim, int nH);

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

	/* Counting variables */
	int N = mxGetM(prhs[0]); /* number of data points */
	int dim = mxGetN(prhs[0]);
	int M = mxGetN(prhs[10]); /* number of grid points */
	int nH = mxGetNumberOfElements(prhs[3]); /* number of hyperplanes */
	double *gradA, *gradB, *TermA, *TermB;

	plhs[0] = mxCreateDoubleMatrix(nH*(dim+1),1,mxREAL);
    gradA = mxGetPr(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(nH*(dim+1),1,mxREAL);
    gradB = mxGetPr(plhs[1]);
	plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
	TermA = mxGetPr(plhs[2]);
	plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL);
	TermB = mxGetPr(plhs[3]);

	/* 1-D */
	if (M==0) {
		M = mxGetN(prhs[10]);
	}
	calcGradFullC(gradA,gradB,influence,TermA,TermB,X,XW,grid,YIdx,a,b,gamma,weight,delta,N,M,dim,nH); 
}
