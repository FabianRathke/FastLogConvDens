#include <mex.h>
#include <mat.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <omp.h>

extern void setGridDensity(double *box, int dim, int sparseGrid, int *N, int *M, double **grid, double* weight, double ratio, int minGridSize);
extern void makeGridC(double *X, unsigned short int **YIdx, unsigned short int **XToBox, int **numPointsPerBox, double **boxEvalPoints, double *ACVH, double *bCVH, double *box, int *lenY, int *numBoxes, int dim, int lenCVH, int N, int M, int NX);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    double *X = (double*)mxGetData(prhs[0]);
    double *box  = mxGetData(prhs[1]);
    double *ACVH  = mxGetData(prhs[2]);
    double *bCVH  = mxGetData(prhs[3]);
    double ratio = mxGetScalar(prhs[4]);
    int minGridSize = (int) mxGetScalar(prhs[5]);

    int n = mxGetM(prhs[0]); /* number of data points */
    int dim = mxGetN(prhs[0]);
    int lenCVH = mxGetNumberOfElements(prhs[3]);

    // create the integration grid
    int lenY, numBoxes = 0;
    int *numPointsPerBox; unsigned short int *YIdx, *XToBox; double *boxEvalPoints;

    // obtain grid density params
    int NGrid, MGrid;
    double weight = 0;
    double *grid = NULL;
    setGridDensity(box,dim,1,&NGrid,&MGrid,&grid,&weight,ratio, minGridSize);

    printf("Obtain grid for N = %d and M = %d\n",NGrid,MGrid);
    makeGridC(X,&YIdx,&XToBox,&numPointsPerBox,&boxEvalPoints,ACVH,bCVH,box,&lenY,&numBoxes,dim,lenCVH,NGrid,MGrid,n);
    printf("Obtained grid with %d points and %d boxes\n",lenY,numBoxes);

    // copy values --> return values to matlab in mex file
    plhs[0] = mxCreateNumericMatrix(dim,lenY,mxUINT16_CLASS,mxREAL);
    memcpy(mxGetPr(plhs[0]),YIdx,lenY*dim*sizeof(unsigned short int));

    plhs[1] = mxCreateNumericMatrix(NGrid*MGrid,dim,mxDOUBLE_CLASS,mxREAL);
    memcpy(mxGetPr(plhs[1]),grid,NGrid*MGrid*dim*sizeof(double));

    plhs[2] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS, mxREAL);
    memcpy(mxGetPr(plhs[2]),&weight,sizeof(double));

	plhs[3] = mxCreateNumericMatrix(pow(NGrid,2),1,mxINT32_CLASS, mxREAL);
    memcpy(mxGetPr(plhs[3]),numPointsPerBox,sizeof(int)*pow(NGrid,2));
}
