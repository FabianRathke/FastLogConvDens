#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mat.h>

extern void calcGradFullAVXC(float* gradA, float* gradB, double* influence, float* TermA, float* TermB, float* X, float* XW, float* grid, unsigned short int* YIdx, float* a, float* b, float gamma, float weight, float* delta, int N, int M, int dim, int nH);
extern void calcGradFloatC(float* gradA, float* gradB, double* influence, float* TermA, float* TermB, float* X, float* XW, float* grid, unsigned short int* YIdx, float* a, float* b, float gamma, float weight, float* delta, int N, int NIter, int M, int dim, int nH);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Matlab input variables */
	float *X = (float *) mxGetData(prhs[0]);
	float *grid = (float *) mxGetData(prhs[1]);
	float *a = (float *) mxGetData(prhs[2]);
	float *b = (float *) mxGetData(prhs[3]);
	float gamma = mxGetScalar(prhs[4]);
	float weight = mxGetScalar(prhs[5]);
	float *delta = (float *) mxGetData(prhs[6]);
	double *influence = (double *) mxGetData(prhs[7]);
	float *XW = (float *) mxGetData(prhs[8]); /* Weight vector for X */
	int n = (int) mxGetScalar(prhs[9]); /* grid dimensions */
	unsigned short int* YIdx = (unsigned short int*) mxGetData(prhs[10]);

    /*const char *file = "/export/home/frathke/Documents/Code/MyCode/LogConcave/code/cudaWorkspace_5_2500.mat";
    MATFile *pmat;

    pmat = matOpen(file, "r");
    if (pmat == NULL) {
        printf("Error opening file %s\n", file);
    }*/

    /* Matlab input variables */
    /*float *X = (float*)mxGetData(matGetVariable(pmat,"XFloat"));
    float *a = (float*)mxGetData(matGetVariable(pmat,"aFloatTrans"));
    float *b = (float*)mxGetData(matGetVariable(pmat,"bFloat"));
    float gamma = (float) mxGetScalar(matGetVariable(pmat,"gamma"));
    float *XW = (float *) mxGetData(matGetVariable(pmat,"sWFloat"));
    float weight = (float) mxGetScalar(matGetVariable(pmat,"weight"));
    float *grid = (float*)mxGetData(matGetVariable(pmat,"gridFloat"));
    float *delta = (float*)mxGetData(matGetVariable(pmat,"deltaFloat"));
    unsigned short int* YIdx = (unsigned short int*) mxGetData(matGetVariable(pmat,"yIdx"));*/

	/* Counting variables */
	int N = mxGetM(prhs[0]); /* number of data points */
	int dim = mxGetN(prhs[0]);
	int M = mxGetN(prhs[10]); /* number of grid points */
	int nH = mxGetNumberOfElements(prhs[3]); /* number of hyperplanes */
	float *gradA, *gradB, *TermA, *TermB;
     
	//printf("%d samples, %d grid points of dim %d, nH: %d\n",N,M,dim,nH);

    const int dimsA[] = {nH*(dim+1)};
    const int dimsB[] = {1};

    plhs[0] = mxCreateNumericArray(1,dimsA,mxSINGLE_CLASS,mxREAL);
    gradA = (float *) mxGetData(plhs[0]);
    plhs[1] = mxCreateNumericArray(1,dimsA,mxSINGLE_CLASS,mxREAL);
    gradB = (float *) mxGetData(plhs[1]);
    plhs[2] = mxCreateNumericArray(1,dimsB,mxSINGLE_CLASS,mxREAL);
    TermA = (float *) mxGetData(plhs[2]);
    plhs[3] = mxCreateNumericArray(1,dimsB,mxSINGLE_CLASS,mxREAL);
    TermB = (float *) mxGetData(plhs[3]);

	*TermA = 0; *TermB = 0;
	for (int i = 0; i < nH; i++) {
		influence[i] = 0;
	}
	int modN, modM;
	modN = N%8; modM = M%8;
	
  	calcGradFullAVXC(gradA,gradB,influence,TermA,TermB,X,XW,grid,YIdx,a,b,gamma,weight,delta,N,M,dim,nH);
//	calcGradFloatC(gradA,gradB,influence,TermA,TermB,X,XW,grid,YIdx,a,b,gamma,weight,delta,N,M,dim,nH);

	calcGradFloatC(gradA,gradB,influence,TermA,TermB,X + N - modN,XW + N - modN,grid,YIdx + (M - modM)*dim,a,b,gamma,weight,delta,N,modN,modM,dim,nH);
}
