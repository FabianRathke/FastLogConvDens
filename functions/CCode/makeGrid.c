#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

extern void makeGridC(double *X, unsigned short int **YIdx, unsigned short int **XToBox, int **numPointsPerBox, double **boxEvalPoints, double *ACVH, double *bCVH, double *box, int **lenY, int **numBoxes, int dim, int lenCVH, int N, int M, int NX, double *sparseDelta);
extern void setGridDensity(double *box, int dim, int sparseGrid, int *N, int *M, double **grid, double* weight, int* gridSize);
extern void makeGridMidpoint(double* box, double* grid, int N, int dim);

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
    int *lenY, *numBoxes, *numPointsPerBox; unsigned short int *YIdx, *XToBox; double *boxEvalPoints;

	if (mxGetM(prhs[2]) != lenCVH) { /* ACVH needs to be in row-wise layout --> one hyperplane per row */
        mexErrMsgTxt("Transpose argument 3.\n");
    }

	int *NTest = malloc(sizeof(int));
	int *MTest = malloc(sizeof(int));
	int *gridSize = malloc(sizeof(int));
	double *weightTest = malloc(sizeof(double));
	double *grid = NULL;
	*NTest = 2; *MTest = 2;
	setGridDensity(box,dim,0,NTest, MTest,&grid,weightTest,gridSize);
	printf("Dim choosen: %d x %d\n",*NTest, *MTest);

	for (int j = 0; j < dim; j++) {
		for (int i = 0; i < *NTest*(*MTest); i++) {
			printf("%.3f, ",grid[i + j*(*NTest)*(*MTest)]);
		}
		printf("\n");
	}
	printf("%.4f\n",*weightTest);
	free(NTest); free(MTest); free(gridSize); free(weightTest); free(grid);	
	makeGridC(X,&YIdx,&XToBox,&numPointsPerBox,&boxEvalPoints,ACVH,bCVH,box,&lenY,&numBoxes,dim,lenCVH,N,M,NX,sparseDelta);

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
	free(boxEvalPoints); free(lenY); free(numBoxes);
}
