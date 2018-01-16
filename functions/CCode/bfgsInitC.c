#include <mex.h>
#include <mat.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

extern void setGridDensity(double *box, int dim, int sparseGrid, int *N, int *M, double **grid, double* weight);
extern void makeGridC(double *X, unsigned short int **YIdx, unsigned short int **XToBox, int **numPointsPerBox, double **boxEvalPoints, double *ACVH, double *bCVH, double *box, int *lenY, int *numBoxes, int dim, int lenCVH, int N, int M, int NX);
extern void calcGradFullAVXC(float* gradA, float* gradB, double* influence, float* TermA, float* TermB, float* X, float* XW, float* grid, unsigned short int* YIdx, float* a, float* b, float gamma, float weight, float* delta, int N, int M, int dim, int nH);
extern void calcGradFloatC(float* gradA, float* gradB, double* influence, float* TermA, float* TermB, float* X, float* XW, float* grid, unsigned short int* YIdx, float* a, float* b, float gamma, float weight, float* delta, int N, int NIter, int M, int dim, int nH);
extern void CNS(double* s_k, double *y_k, double *sy, double *syInv, double step, double *grad, double *gradOld, double *newtonStep, int numIter, int activeCol, int nH, int m);

void unzipParams(float* params, float* a, float* b, int dim, int nH) {
	int i,j;
	// transpose operation
	for (i=0; i < dim; i++) {
	   for (j=0; j < nH; j++) {
		   a[j*dim + i] = params[j + i*nH];
	   }
	}
	for (i=0; i < nH; i++) {
		b[i] = params[dim*nH+i];
	}	
	//b = params + dim*nH;
}

double calcLambdaSq(double* grad, double* newtonStep, int dim, int nH) {
	double lambdaSq = 0;
	int i;
	for (i=0; i < nH*(dim+1); i++) {
		lambdaSq += grad[i]*-newtonStep[i];
	}
	return lambdaSq;
}	


void sumGrad(double* grad, float* gradA, float* gradB, int n) {
	int i;
	for (i=0; i < n; i++) {
		grad[i] = (double) (gradA[i] + gradB[i]);
	}
}

void copyVector(double* dest, double* source, int n, int switchSign) {
	int i;
	if (switchSign == 1) {
		for (i=0; i < n; i++) {
			dest[i] = -source[i];
		}
	} else {
		for (i=0; i < n; i++) {
			dest[i] = source[i];
		}
	}
}	

void calcGradFloatAVXCaller(float *X, float* XW, float *grid, float* a, float* b, float gamma, float weight, float* delta, int n, int dim, int nH, int M, float* gradA, float* gradB, float* TermA, float* TermB, double* influence, unsigned short int* YIdx) {
    
    for (int i = 0; i < nH; i++) {
        influence[i] = 0;
    }
    int modn, modM;
    modn = n%8; modM = M%8;
	
	// set gradients to zero
	memset(gradA,0,nH*(dim+1)*sizeof(float));
	memset(gradB,0,nH*(dim+1)*sizeof(float));
	// set TermA and TermB to zero
	*TermA = 0; *TermB = 0;

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
void newtonBFGSLInitC(float* X,  float* XW, double* box, float* params, int dim, int lenP, int n, double* ACVH, double* bCVH, int lenCVH, double intEps, double lambdaSqEps, double* logLike) {

	int i;
	
	// number of hyperplanes
	int nH  = (int) lenP/(dim+1);

	// create the integration grid
    int lenY, numBoxes = 0;
	int *numPointsPerBox; unsigned short int *YIdx, *XToBox; double *boxEvalPoints;

	// obtain grid density params
	int NGrid, MGrid;
    double weight = 0; 
    double *grid = NULL;
    setGridDensity(box,dim,1,&NGrid,&MGrid,&grid,&weight);
	
	float *delta = malloc(dim*sizeof(float));
	for (i=0; i < dim; i++) {
		delta[i] = grid[NGrid*MGrid*i+1] - grid[NGrid*MGrid*i];
	}
	double *XDouble = malloc(n*dim*sizeof(double));
	for (i=0; i < n*dim; i++) {
		XDouble[i] = (double) X[i];
	}
	//printf("Obtain grid for N = %d and M = %d\n",NGrid,MGrid);
	makeGridC(XDouble,&YIdx,&XToBox,&numPointsPerBox,&boxEvalPoints,ACVH,bCVH,box,&lenY,&numBoxes,dim,lenCVH,NGrid,MGrid,n);
	//printf("Obtained grid with %d points\n",lenY);
	
	// only the first entry in each dimension is required
	float *gridFloat = malloc(dim*sizeof(float));
	for (i=0; i < dim; i++) {
		gridFloat[i] = grid[i*NGrid*MGrid];
	}
	// two points for a and b: slope and bias of hyperplanes
	float *a = malloc(nH*dim*sizeof(float));
	float *aNew = malloc(nH*dim*sizeof(float));
	float *b = malloc(nH*sizeof(float));
	float *bNew = malloc(nH*sizeof(float));

	unzipParams(params,a,b,dim,nH);

	double *influence = malloc(nH*sizeof(double));
	double alpha = 1e-4, beta = 0.1;
	float gamma = 1;

	double *grad = malloc(nH*(dim+1)*sizeof(double));
	double *gradOld = malloc(nH*(dim+1)*sizeof(double));
	double *newtonStep = malloc(nH*(dim+1)*sizeof(double));
	float *paramsNew = malloc(lenP*sizeof(float));
	float *gradA = calloc(nH*(dim+1),sizeof(float));
	float *gradB = calloc(nH*(dim+1),sizeof(float));
	float *TermA = calloc(1,sizeof(float));
	float *TermB = calloc(1,sizeof(float));
	float TermAOld, TermBOld, funcVal, funcValStep;
	float lastStep;
	calcGradFloatAVXCaller(X, XW, gridFloat, a, b, gamma, weight, delta, n, dim, nH, lenY, gradA, gradB, TermA, TermB, influence, YIdx);
	sumGrad(grad,gradA,gradB,nH*(dim+1));
	
	copyVector(newtonStep,grad,nH*(dim+1),1);
	// LBFGS params
	int m = 10;
	double* s_k = calloc(lenP*m,sizeof(double));
	double* y_k = calloc(lenP*m,sizeof(double));
	double* sy = calloc(m,sizeof(double));
	double* syInv = calloc(m,sizeof(double));
	double lambdaSq, step;
	int iter, numIter;
	int activeCol = 0;
	// start the main iteration
	for (iter = 0; iter < 1e4; iter++) {
		lambdaSq = calcLambdaSq(grad,newtonStep,dim,nH);
		if (lambdaSq < 0 || lambdaSq > 1e5) {
			for (i=0; i < nH*(dim+1); i++) {
				newtonStep[i] = -grad[i];
			}
			lambdaSq = calcLambdaSq(grad,newtonStep,dim,nH);
		}

		step = 1;
		// objective function value before the step
		TermAOld = *TermA; TermBOld = *TermB; funcVal = TermAOld + TermBOld; copyVector(gradOld,grad,nH*(dim+1),0);
		// new parameters
		for (i=0; i < lenP; i++) { paramsNew[i] = params[i] + (float) newtonStep[i]; }
		unzipParams(paramsNew,aNew,bNew,dim,nH);
		// calculate gradient and objective function value
		calcGradFloatAVXCaller(X, XW, gridFloat, aNew, bNew, gamma, weight, delta, n, dim, nH, lenY, gradA, gradB, TermA, TermB, influence, YIdx);
		sumGrad(grad,gradA,gradB,(dim+1)*nH);
		funcValStep = *TermA + *TermB;

		while (isnan(funcValStep) || isinf(funcValStep) || funcValStep > funcVal - step*alpha*lambdaSq) {
			if (step < 1e-9) {
				break;
			}
			step = beta*step;
			for (i=0; i < lenP; i++) {
				paramsNew[i] = params[i] + (float) (newtonStep[i]*step);
			}
			unzipParams(paramsNew,aNew,bNew,dim,nH);

			calcGradFloatAVXCaller(X, XW, gridFloat, aNew, bNew, gamma, weight, delta, n, dim, nH, lenY, gradA, gradB, TermA, TermB, influence, YIdx);
			sumGrad(grad,gradA,gradB,(dim+1)*nH);
			funcValStep = *TermA + *TermB;
		}
		lastStep = funcVal - funcValStep;

		//printf("%d: %.5f (%.4f, %.5f, %d) \t (lambdaSq: %.4e, t: %.0e, Step: %.4e)\n",iter,funcValStep,-*TermA*n,*TermB,nH,lambdaSq,step,lastStep);
		for (i=0; i < lenP; i++) { params[i] = paramsNew[i]; }
		
		if (fabs(1-*TermB) < intEps && lastStep < lambdaSqEps && iter > 10) {
			break;
		}
	
		// min([m,iter,length(b)]) --> C indexing of iter is one less than matlab --> +1
		numIter = m < iter+1 ? m : iter+1;
		numIter = nH < numIter ? nH : numIter;
		CNS(s_k,y_k,sy,syInv,step,grad,gradOld,newtonStep,numIter,activeCol,lenP,m);
		activeCol++; 
    	if (activeCol >= m) {
        	activeCol = 0;
		}
	}
	logLike[0] = funcValStep;
	free(gradA); free(gradB); free(a); free(b); free(XDouble); free(delta); free(gridFloat); free(s_k); free(y_k); free(sy); free(syInv);
	free(grad); free(gradOld); free(newtonStep); free(paramsNew);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

   /* Matlab input variables */
    float *X = (float*)mxGetData(prhs[0]);
    float *XW = (float *) mxGetData(prhs[1]); /* Weight vector for X */
    float *params = (float*)mxGetData(prhs[2]);
    double *box  = mxGetData(prhs[3]);
    double *ACVH  = mxGetData(prhs[4]);
    double *bCVH  = mxGetData(prhs[5]);
    double *logLike  = mxGetData(prhs[6]);

	int n = mxGetM(prhs[0]); /* number of data points */
	int dim = mxGetN(prhs[0]);
	int lenP = mxGetNumberOfElements(prhs[2]); /* number of hyperplanes */
	int lenCVH = mxGetNumberOfElements(prhs[5]);

	double intEps = 1e-3;
	double lambdaSqEps = 1e-6; // for the initialization

	//printf("%d samples in dimension %d\n", n,dim);
	//printf("%d params, %d faces of conv(X)\n", lenP,lenCVH);

	newtonBFGSLInitC(X, XW, box, params, dim, lenP, n, ACVH, bCVH, lenCVH, intEps, lambdaSqEps, logLike);
	printf("%.4f\n",logLike[0]);
}
