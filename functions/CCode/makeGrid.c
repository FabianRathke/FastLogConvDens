#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

extern void makeGridC(double *X, double *sparseGrid, unsigned short int **YIdx, unsigned short int **XToBox, unsigned short int **XToBoxOuter, int **numPointsPerBox, double **boxEvalPoints, double *subGrid, int *subGridIdx, double *ACVH, double *bCVH, double *box, int **lenY, int **numBoxes, int dim, int lenCVH, int N, int M, int numGridPoints, int NX, double *sparseDelta);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Input variables */
	double *sparseGrid = mxGetPr(prhs[0]);
	double *box = mxGetPr(prhs[1]); // bounding box: max(X)-min(X)
	double *ACVH = mxGetPr(prhs[2]);
	double *bCVH = mxGetPr(prhs[3]);
	int N = (int) mxGetScalar(prhs[4]);
	int M = (int) mxGetScalar(prhs[5]);
	int dim = (int) mxGetScalar(prhs[6]);
	double *X = mxGetPr(prhs[7]);
	double *sparseDelta = mxGetPr(prhs[8]);

    int lenCVH = mxGetNumberOfElements(prhs[3]);
	int NX = mxGetM(prhs[7]); /* number of data points */
    int *lenY, *numBoxes; unsigned short int *YIdx, *XToBox, *XToBoxOuter; int *numPointsPerBox; double *boxEvalPoints;

	double *subGrid = malloc(dim*M*sizeof(double));
    int *subGridIdx = malloc(dim*pow(M,dim)*sizeof(int));
	int numGridPoints;

	if (mxGetM(prhs[2]) != lenCVH) { /* ACVH needs to be in row-wise layout --> one hyperplane per row */
        mexErrMsgTxt("Transpose argument 3.\n");
    }

   	numGridPoints = pow(N+1,dim);

	makeGridC(X,sparseGrid,&YIdx,&XToBox,&XToBoxOuter,&numPointsPerBox,&boxEvalPoints,subGrid,subGridIdx,ACVH,bCVH,box,&lenY,&numBoxes,dim,lenCVH,N,M,numGridPoints,NX,sparseDelta);

	plhs[0] = mxCreateNumericMatrix(dim, *lenY, mxUINT16_CLASS, mxREAL);
	memcpy(mxGetPr(plhs[0]),YIdx,dim*(*lenY)*sizeof(unsigned short int));
	free(YIdx);

	if (nlhs > 1) {
		plhs[1] = mxCreateNumericMatrix(1, NX, mxUINT16_CLASS, mxREAL);
		memcpy(mxGetPr(plhs[1]),XToBox,NX*sizeof(unsigned short int));
	}
	free(XToBox);

	if (nlhs > 2) {
		plhs[2] = mxCreateNumericMatrix(1,*numBoxes+1,mxINT32_CLASS, mxREAL);
		memcpy(mxGetPr(plhs[2]),numPointsPerBox,(*numBoxes+1)*sizeof(int));
	}
	free(numPointsPerBox);

	if (nlhs > 3) {
		plhs[3] = mxCreateDoubleMatrix(dim,*numBoxes*3,mxREAL);
		memcpy(mxGetPr(plhs[3]),boxEvalPoints,*numBoxes*dim*3*sizeof(double));
	}
	free(boxEvalPoints);
	free(subGrid);
	free(subGridIdx);
	free(XToBoxOuter);

	free(lenY); free(numBoxes);
}
