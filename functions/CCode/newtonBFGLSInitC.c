#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void unzipParams(float* params, float* a, float* b, int dim, int nH) {
	int i,j;
	// transpose operation
	for (i=0; i < dim; i++) {
	   for (j=0; j < nH; j++) {
		   a[j*dim + i] = params[j + i*nH];
	   }
	}	   
	b = paramsInit + dim*nH;
}


void calcGradFloatAVXCaller(float *X, float* XW, float *grid, float* a, float* b, float gamma, float weight, float* delta, int N, int dim, int nH, int M, float* gradA, float* gradB, float* TermA, *float TermB, double* influence, unsigned short int* YIdx) {
    
    const int dimsA[] = {nH*(dim+1)};
    const int dimsB[] = {1};

    for (int i = 0; i < nH; i++) {
        influence[i] = 0;
    }
    int modN, modM;
    modN = N%8; modM = M%8;

	// perform AVX for most entries except the one after the last devisor of 8
    calcGradFullAVXC(gradA,gradB,influence,TermA,TermB,X,XW,grid,YIdx,a,b,gamma,weight,delta,N,M,dim,nH);
    calcGradFloatC(gradA,gradB,influence,TermA,TermB,X + n - modN,XW + n - modN,grid,YIdx + (M - modM)*dim,a,b,gamma,weight,delta,N,modN,modM,dim,nH);
}

/* newtonBFGLSInitC
 *
 * Input: 	float* X			the samples
 * 			float* XW			sample weights
 * 			float* paramsInit	initial parameter vector
 * 			int dim				dimension of X
 * 			int lenP			size of paramsInit
 * 			int n				number of samples
 * 			int M				number of grid points
 * */
void newtonBFGSLInitC(float* X,  float* XW, float* params, int dim, int lenP, int n, int M, double* ACVH, double* bCVH, int lenCVH) {

	// number of hyperplanes
	int nH  = (int) lenP/lenP;

	// create the integration grid
    int *lenY, *numBoxes, *numPointsPerBox; unsigned short int *YIdx, *XToBox; double *boxEvalPoints;

	int NGrid, MGrid;
    double weight; 
    double *grid = NULL;
    setGridDensity(box,dim,0,&NGrid,&MGrid,&grid,&weight,gridSize);

	float *delta = malloc(dim*sizeof(float));
	for (i=0; i < dim; i++) {
		delta[i] = grid[(*NGrid)*(*MGrid)*i+1] - grid[(*NGrid)*(*MGrid)*i];
	}
	makeGridC(X,&YIdx,&XToBox,&numPointsPerBox,&boxEvalPoints,ACVH,bCVH,box,&lenY,&numBoxes,dim,lenCVH,NGrid,MGrid,n);


	// two points for a and b: slope and bias of hyperplanes
	float *a = malloc(nH*dim*sizeof(float));
	float *b = NULL;
	unzipParams(params,a,b);

	double *influence = malloc(nH*sizeof(double));
	double alpha = 0.0001, beta = 0.1;
	float gamma = 1;

	float *gradA = malloc(nH*(dim+1)*sizeof(float));
	float *gradB = malloc(nH*(dim+1)*sizeof(float));
	float *TermA = malloc(sizeof(float));
	float *TermB = malloc(sizeof(float));
	calcGradFloatAVXCaller(X,XW,grid, float* a, float* b, float gamma, float weight, float* delta, n, dim, nH, lenY, gradA, gradB, TermA, TermB, influence, YIdx) {
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Matlab input variables */
	float *X = (float*) mxGetData(prhs[0]);
	float *grid = (float*) mxGetData(prhs[1]);
	float *a = (float*) mxGetData(prhs[2]); // transposed version
	float *b = (float*) mxGetData(prhs[3]);
	float gamma = mxGetScalar(prhs[4]);
	float weight = mxGetScalar(prhs[5]);
	float *delta = (float*) mxGetData(prhs[6]);
	double *influence =  mxGetPr(prhs[7]);
	float *XW = (float*) mxGetData(prhs[8]); /* Weight vector for X */
	int n = (int) mxGetScalar(prhs[9]); /* grid dimensions */
	unsigned short int* YIdx = (unsigned short int*) mxGetData(prhs[10]);
	int *numPointsPerBox = (int*)mxGetData(prhs[11]);
	float *boxEvalPoints = (float*) mxGetData(prhs[12]);
	unsigned short int *XToBox = (unsigned short int*)mxGetData(prhs[13]);
	int MBox = (int) mxGetScalar(prhs[14]);
	float *evalFunc = (float*) mxGetData(prhs[15]);

	/* Counting variables */
	int N = mxGetM(prhs[0]); /* number of data points */
	int dim = mxGetN(prhs[0]);
	int M = mxGetN(prhs[10]); /* number of grid points */
	int nH = mxGetNumberOfElements(prhs[3]); /* number of hyperplanes */
	int numBoxes;	
	float *gradA, *gradB, *TermA, *TermB;

	plhs[0] = mxCreateNumericMatrix(nH*(dim+1),1,mxSINGLE_CLASS,mxREAL);
    gradA = (float*) mxGetData(plhs[0]);
	plhs[1] = mxCreateNumericMatrix(nH*(dim+1),1,mxSINGLE_CLASS,mxREAL);
    gradB = (float*) mxGetData(plhs[1]);
	plhs[2] = mxCreateNumericMatrix(1,1,mxSINGLE_CLASS,mxREAL);
	TermA = (float*) mxGetData(plhs[2]);
	plhs[3] = mxCreateNumericMatrix(1,1,mxSINGLE_CLASS,mxREAL);
	TermB = (float*) mxGetData(plhs[3]);

	numBoxes = mxGetNumberOfElements(prhs[11])-1; /* number of active boxes */
	calcGradAVXC(gradA,gradB,influence,TermA,TermB,X,XW,grid,YIdx,numPointsPerBox,boxEvalPoints,XToBox,numBoxes,a,b,gamma,weight,delta,N,M,dim,nH,MBox,evalFunc);
}
