#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

extern void setGridDensity(double *box, int dim, int sparseGrid, int *N, int *M, double **grid, double* weight);
extern void makeGridC(double *X, unsigned short int **YIdx, unsigned short int **XToBox, int **numPointsPerBox, double **boxEvalPoints, double *ACVH, double *bCVH, double *box, int *lenY, int *numBoxes, int dim, int lenCVH, int N, int M, int NX);
extern void calcGradFullAVXC(float* gradA, float* gradB, double* influence, float* TermA, float* TermB, float* X, float* XW, float* grid, unsigned short int* YIdx, float* a, float* b, float gamma, float weight, float* delta, int N, int M, int dim, int nH);
extern void calcGradFloatC(float* gradA, float* gradB, double* influence, float* TermA, float* TermB, float* X, float* XW, float* grid, unsigned short int* YIdx, float* a, float* b, float gamma, float weight, float* delta, int N, int NIter, int M, int dim, int nH);

void unzipParams(float* params, float* a, float* b, int dim, int nH) {
	int i,j;
	// transpose operation
	for (i=0; i < dim; i++) {
	   for (j=0; j < nH; j++) {
		   a[j*dim + i] = params[j + i*nH];
	   }
	}	   
	b = params + dim*nH;
}


void calcGradFloatAVXCaller(float *X, float* XW, float *grid, float* a, float* b, float gamma, float weight, float* delta, int n, int dim, int nH, int M, float* gradA, float* gradB, float* TermA, float* TermB, double* influence, unsigned short int* YIdx) {
    
    for (int i = 0; i < nH; i++) {
        influence[i] = 0;
    }
    int modn, modM;
    modn = n%8; modM = M%8;

	// perform AVX for most entries except the one after the last devisor of 8
    calcGradFullAVXC(gradA,gradB,influence,TermA,TermB,X,XW,grid,YIdx,a,b,gamma,weight,delta,n,M,dim,nH);
    calcGradFloatC(gradA,gradB,influence,TermA,TermB,X + n - modn,XW + n - modn,grid,YIdx + (M - modM)*dim,a,b,gamma,weight,delta,n,modn,modM,dim,nH);
}

/* newtonBFGLSInitC
 *
 * Input: 	float* X			the samples
 * 			float* XW			sample weights
 * 			float* paramsInit	initial parameter vector
 * 			int dim				dimension of X
 * 			int lenP			size of paramsInit
 * 			int n				number of samples
 * */
void newtonBFGSLInitC(float* X,  float* XW, double* box, float* params, int dim, int lenP, int n, double* ACVH, double* bCVH, int lenCVH) {

	int i,j,k;
	
	// number of hyperplanes
	int nH  = (int) lenP/(dim+1);

	// create the integration grid
    int lenY, numBoxes;
	int *numPointsPerBox; unsigned short int *YIdx, *XToBox; double *boxEvalPoints;

	// obtain grid density params
	int NGrid, MGrid;
    double weight; 
    double *grid = NULL;
    setGridDensity(box,dim,0,&NGrid,&MGrid,&grid,&weight);

	float *delta = malloc(dim*sizeof(float));
	for (i=0; i < dim; i++) {
		delta[i] = grid[NGrid*MGrid*i+1] - grid[NGrid*MGrid*i];
	}
	double *XDouble = malloc(n*dim*sizeof(double));
	for (i=0; i < n*dim; i++) {
		XDouble[i] = (double) X[i];
	}
	makeGridC(XDouble,&YIdx,&XToBox,&numPointsPerBox,&boxEvalPoints,ACVH,bCVH,box,&lenY,&numBoxes,dim,lenCVH,NGrid,MGrid,n);

	float *gridFloat = malloc(NGrid*MGrid*dim*sizeof(float));
	for (i=0; i < NGrid*MGrid*dim; i++) {
		gridFloat[i] = grid[i];
	}
	// two points for a and b: slope and bias of hyperplanes
	float *a = malloc(nH*dim*sizeof(float));
	float *b = NULL;
	unzipParams(params,a,b,dim,nH);

	double *influence = malloc(nH*sizeof(double));
	double alpha = 0.0001, beta = 0.1;
	float gamma = 1;

	float *gradA = malloc(nH*(dim+1)*sizeof(float));
	float *gradB = malloc(nH*(dim+1)*sizeof(float));
	float *TermA = malloc(sizeof(float));
	float *TermB = malloc(sizeof(float));
	calcGradFloatAVXCaller(X, XW, gridFloat, a, b, gamma, weight, delta, n, dim, nH, lenY, gradA, gradB, TermA, TermB, influence, YIdx);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Matlab input variables */
	float *X = (float*) mxGetData(prhs[0]);
	float *XW = (float*) mxGetData(prhs[1]); /* Weight vector for X */
	float *params = (float*) mxGetData(prhs[2]);
    double *box = mxGetPr(prhs[3]); // bounding box: max(X)-min(X)
	double *ACVH = mxGetPr(prhs[4]);
	double *bCVH = mxGetPr(prhs[5]);

	int n = mxGetM(prhs[0]); /* number of data points */
	int dim = mxGetN(prhs[0]);
	int lenP = mxGetNumberOfElements(prhs[3]); /* number of hyperplanes */
	int lenCVH = mxGetNumberOfElements(prhs[5]);

	newtonBFGSLInitC(X, XW, box, params, dim, lenP, n, ACVH, bCVH, lenCVH);
}
