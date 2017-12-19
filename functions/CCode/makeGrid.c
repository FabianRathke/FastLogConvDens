#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "headers.h"

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
    unsigned char *aH;  int *lenY, *numBoxes; unsigned short int *YIdx, *YIdxSub, *gridToBox, *XToBox, *XToBoxOuter, *boxIDs; int *numPointsPerBox; double *boxEvalPoints;

	double *subGrid = malloc(dim*M*sizeof(double));
    int *subGridIdx = malloc(dim*pow(M,dim)*sizeof(int));
	int numGridPoints;

	if (mxGetM(prhs[2]) != lenCVH) { /* ACVH needs to be in row-wise layout --> one hyperplane per row */
        mexErrMsgTxt("Transpose argument 3.\n");
    }

   	numGridPoints = pow(N+1,dim);

	makeGridC(X,sparseGrid,&aH,&YIdxSub,&YIdx,&gridToBox,&XToBox,&XToBoxOuter,&numPointsPerBox,&boxEvalPoints,&boxIDs,subGrid,subGridIdx,ACVH,bCVH,box,&lenY,&numBoxes,dim,lenCVH,N,M,numGridPoints,NX,sparseDelta);

	plhs[0] = mxCreateNumericMatrix(dim, *lenY, mxUINT16_CLASS, mxREAL);
	memcpy(mxGetPr(plhs[0]),YIdx,dim*(*lenY)*sizeof(unsigned short int));
	free(YIdx);
	if (nlhs > 1) {
		plhs[1] = mxCreateNumericMatrix(1, *lenY, mxUINT16_CLASS, mxREAL);
		memcpy(mxGetPr(plhs[1]),gridToBox,(*lenY)*sizeof(unsigned short int));
	}
	free(gridToBox);

	if (nlhs > 2) {
		plhs[2] = mxCreateNumericMatrix(1, NX, mxUINT16_CLASS, mxREAL);
		memcpy(mxGetPr(plhs[2]),XToBox,NX*sizeof(unsigned short int));
	}
	free(XToBox);

	if (nlhs > 3) {
		plhs[3] = mxCreateNumericMatrix(1,*numBoxes+1,mxINT32_CLASS, mxREAL);
		memcpy(mxGetPr(plhs[3]),numPointsPerBox,(*numBoxes+1)*sizeof(int));
	}
	free(numPointsPerBox);

	if (nlhs > 4) {
		plhs[4] = mxCreateDoubleMatrix(dim,*numBoxes*3,mxREAL);
		memcpy(mxGetPr(plhs[4]),boxEvalPoints,*numBoxes*dim*3*sizeof(double));
	}
	free(boxEvalPoints);

	if (nlhs > 5) {
		plhs[5] = mxCreateNumericMatrix(1, *lenY, mxUINT16_CLASS, mxREAL);
		memcpy(mxGetPr(plhs[5]),YIdxSub,(*lenY)*sizeof(unsigned short int));
	}
	free(YIdxSub);

	if (nlhs > 6) {
		plhs[6] = mxCreateDoubleMatrix(M, dim, mxREAL);
		memcpy(mxGetPr(plhs[6]),subGrid,dim*M*sizeof(double));
	}
	free(subGrid);

	if (nlhs > 7) {
		plhs[7] = mxCreateNumericMatrix(dim, pow(M,dim), mxINT32_CLASS, mxREAL);
		memcpy(mxGetPr(plhs[7]),subGridIdx,pow(M,dim)*dim*sizeof(int));
	}
	free(subGridIdx);

	if (nlhs > 8) {
		plhs[8] = mxCreateNumericMatrix(1,*numBoxes, mxUINT16_CLASS, mxREAL);
		memcpy(mxGetPr(plhs[8]),boxIDs,*numBoxes*sizeof(unsigned short int));
	}
	free(boxIDs);

	if (nlhs > 9) {
		plhs[9] = mxCreateNumericMatrix(1,NX, mxUINT16_CLASS, mxREAL);
		memcpy(mxGetPr(plhs[9]),XToBoxOuter,NX*sizeof(unsigned short int));
	}
	free(XToBoxOuter);

	if (nlhs > 10) {
		plhs[10] = mxCreateNumericMatrix(lenCVH, *numBoxes, mxUINT8_CLASS, mxREAL);
		memcpy(mxGetPr(plhs[10]),aH,*numBoxes*lenCVH*sizeof(unsigned char));
	}

	free(aH);

	free(lenY); free(numBoxes);
}
