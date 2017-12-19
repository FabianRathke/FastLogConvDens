#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "headers.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Input variables */
	double *a = mxGetPr(prhs[0]);
	double *b = mxGetPr(prhs[1]);
	double gamma = mxGetScalar(prhs[2]);
	double *influence = mxGetPr(prhs[3]);
	double *Y = mxGetPr(prhs[4]);

	/* Counting variables */
	int dim = mxGetM(prhs[4]);
	int M = mxGetN(prhs[4]); /* number of grid points */
	int nH = mxGetNumberOfElements(prhs[1]); /* number of hyperplanes */	
	
	evalInfluenceC(influence,Y,a,b,gamma,M,dim,nH);
}
