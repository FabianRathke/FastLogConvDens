#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "headers.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Matlab input variables */
	double *X = mxGetPr(prhs[0]);
	int n = mxGetScalar(prhs[1]);
	int *T = (int *)mxGetData(prhs[2]);
	double *yT = mxGetPr(prhs[3]);
	int *samplesSimplex = (int *)mxGetData(prhs[4]);

	int N = mxGetM(prhs[0]); /* number of data points */
	int dim = mxGetN(prhs[0]);
	int lenT = mxGetM(prhs[2]);
	double *samples,*samplesEval;

	plhs[0] = mxCreateDoubleMatrix(n*dim,1,mxREAL);
    samples = mxGetPr(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(n,1,mxREAL);
	samplesEval = mxGetPr(plhs[1]);

	samplesLCDC(X,n,T,yT,samplesSimplex,samples,samplesEval,N,dim,lenT);
}
