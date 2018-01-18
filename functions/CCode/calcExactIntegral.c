#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

extern void calcExactIntegralC(double* X, double* y, int* T, int lenT, int lenY, int dim, double* integral, double targetIntegral, double intEps, double* changeB, double* AdSave, double* GdSave);
extern void recalcParamsC(double* X, double* y, int* T, int lenT, int dim, double* aOptNew, double* bOptNew);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* default values */
	int correctIntegral = 1;
	double intEps = pow(10,-6);
	double targetIntegral = 1;
	
	/* Input variables */
	double *X = mxGetPr(prhs[0]);
	double *y = mxGetPr(prhs[1]);
	int *T = (int *)mxGetData(prhs[2]);
	int dim = (int) mxGetScalar(prhs[3]);

	int lenY,lenT;
	/* Output variables */
	double *aOptNew, *bOptNew, *integral, *changeB, *Ad, *Gd;

	/* overwrite some default values depending on user input */
	if (nrhs > 4) {
		correctIntegral = (int) mxGetScalar(prhs[4]);
	}
	if (nrhs > 5) {
		intEps = (double) mxGetScalar(prhs[5]);
	}
	if (nrhs > 6) {
		targetIntegral = (double) mxGetScalar(prhs[6]);
	}


    lenY = mxGetM(prhs[1]) > mxGetN(prhs[1]) ? mxGetM(prhs[1]) : mxGetN(prhs[1]); /* number of supporting points */
    lenT = mxGetN(prhs[2]); /* number of simplices */

    if (mxGetM(prhs[2]) != (dim+1)) { /* each column of T holds one simplex */
        mexErrMsgTxt("Numbers of columns of T have to correspond to d+1 (Argument 3).\n");
    }

    plhs[0] = mxCreateDoubleMatrix(dim,lenT,mxREAL);
	aOptNew = mxGetPr(plhs[0]);

    plhs[1] = mxCreateDoubleMatrix(lenT,1,mxREAL);
	bOptNew = mxGetPr(plhs[1]);

	plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
	integral = mxGetPr(plhs[2]);

	plhs[3] = mxCreateDoubleMatrix(1,1,mxREAL);
	changeB = mxGetPr(plhs[3]);

	plhs[4] = mxCreateDoubleMatrix(lenT,1,mxREAL);
	Ad = mxGetPr(plhs[4]);

	plhs[5] = mxCreateDoubleMatrix(lenT,1,mxREAL);
	Gd = mxGetPr(plhs[5]);


	if (correctIntegral == 1) {
		calcExactIntegralC(X,y,T,lenT,lenY,dim,integral,targetIntegral,intEps,changeB,Ad,Gd);
	}
	recalcParamsC(X,y,T,lenT,dim,aOptNew,bOptNew);
}
