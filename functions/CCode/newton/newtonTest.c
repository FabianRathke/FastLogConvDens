#include <mex.h>
#include <mat.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <omp.h>

extern double cpuSecond();
extern void setGridDensity(double *box, int dim, int sparseGrid, int *N, int *M, double **grid, double* weight);
extern void makeGridC(double *X, unsigned short int **YIdx, unsigned short int **XToBox, int **numPointsPerBox, double **boxEvalPoints, double *ACVH, double *bCVH, double *box, int *lenY, int *numBoxes, int dim, int lenCVH, int N, int M, int NX);
extern void CNS(double* s_k, double *y_k, double *sy, double *syInv, double step, double *grad, double *gradOld, double *newtonStep, int numIter, int activeCol, int nH, int m);
extern void calcGradAVXC(double* gradA, double* gradB, double* influence, double* TermA, double* TermB, float* X, float* XW, float* grid, unsigned short int* YIdx, int *numPointsPerBox, float* boxEvalPoints, unsigned short int *XToBox, int numBoxes, double* a, double* b, float gamma, float weight, float* delta, int N, int M, int dim, int nH, int MBox, float* evalFunc);
extern void calcGradFloatC(float* gradA, float* gradB, double* influence, float* TermA, float* TermB, float* X, float* XW, float* grid, unsigned short int* YIdx, float* a, float* b, float gamma, float weight, float* delta, int N, int NIter, int M, int dim, int nH);

void unzipParams(double *params, double *a, double *b, int dim, int nH) {
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
}

double calcLambdaSq(double *grad, double *newtonStep, int dim, int nH) {
	double lambdaSq = 0;
	int i;
	for (i=0; i < nH*(dim+1); i++) {
		lambdaSq += grad[i]*-newtonStep[i];
	}
	return lambdaSq;
}	


void sumGrad(double *grad, double *gradA, double *gradB, int n) {
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

void resetGradientFloat(double* gradA, double* gradB, double* TermA, double* TermB, int lenP) {
	// set gradients to zero
	memset(gradA,0,lenP*sizeof(double));
	memset(gradB,0,lenP*sizeof(double));
	// set TermA and TermB to zero
	*TermA = 0; *TermB = 0;
}

void resizeArray(double** array, int* keepIdx, int nNew, int n, int dim) {
    // resizes array by only keeping rows marked by keepIdx
    int i,j;
    for (j = 0; j < dim; j++) {
        for (i = 0; i < nNew; i++) {
            (*array)[i + j*nNew] = (*array)[keepIdx[i] + j*n];
        }
    }
    // realloc array; use temporary pointer to check for failure
    double *newArray = realloc(*array, nNew*dim*sizeof(double));
    if (newArray == NULL && nNew > 0) {
        printf("Array reallocation failed\n");
        exit(0);
    }
    *array = newArray;
}

/* newtonBFGLSC
 *
 * Input: 	float* X			the samples
 * 			float* XW			sample weights
 * 			float* paramsInit	initial parameter vector
 * 			int dim				dimension of X
 * 			int lenP			size of paramsInit
 * 			int n				number of samples
 * 			double* ACVH		slopes of hyperplanes of the convex hull of X
 * 			double* bCVH		offset of hyperplanes of the convex hull of X
 * 			int lenCVH			number of faces in the convex hull of X
 * 			double intEps		required accuracy of the integration error
 * 			double lambdaSqEps	minimal progress of the optimization in terms of objective function value
 * 			double cutoff		threshold for removing inactive hyperplanes.
 * */
void newtonBFGSLC(float* X,  float* XW, double* box, float* params_, int dim, int lenP, int n, double* ACVH, double* bCVH, int lenCVH, double intEps, double lambdaSqEps, double cutoff, int verbose) {

	int i;
	double timeA = cpuSecond();
	
	// number of hyperplanes
	int nH  = (int) lenP/(dim+1);

	// create the integration grid
    int lenY, numBoxes = 0;
	int *numPointsPerBox; unsigned short int *YIdx, *XToBox; double *boxEvalPoints; 

	// obtain grid density params
	int NGrid, MGrid;
    double weight = 0; 
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
	//printf("Obtain grid for N = %d and M = %d\n",NGrid,MGrid);
	makeGridC(XDouble,&YIdx,&XToBox,&numPointsPerBox,&boxEvalPoints,ACVH,bCVH,box,&lenY,&numBoxes,dim,lenCVH,NGrid,MGrid,n);
	// printf("Obtained grid with %d points and %d boxes\n",lenY,numBoxes);

	float *boxEvalPointsFloat = malloc(numBoxes*dim*3*sizeof(float));
	for (i=0; i < numBoxes*dim*3; i++) { boxEvalPointsFloat[i] = (float) boxEvalPoints[i]; }
	// only the first entry in each dimension is required
	float *gridFloat = malloc(dim*sizeof(float));
	for (i=0; i < dim; i++) {
		gridFloat[i] = grid[i*NGrid*MGrid];
	}
	// two points for a and b: slope and bias of hyperplanes
	double *a = malloc(nH*dim*sizeof(double));
	double *aNew = malloc(nH*dim*sizeof(double));
	double *b = malloc(nH*sizeof(double));
	double *bNew = malloc(nH*sizeof(double));
	double *params = malloc(lenP*sizeof(double));
	for (i=0; i < lenP; i++) {params[i] = (double) params_[i]; }

	unzipParams(params,a,b,dim,nH);

	double *influence = malloc(nH*sizeof(double));
	double alpha = 1e-4, beta = 0.1;
	float gamma = 1000;

	double *grad = malloc(lenP*sizeof(double));
	double *gradOld = malloc(lenP*sizeof(double));
	double *newtonStep = malloc(lenP*sizeof(double));
	double *paramsNew = malloc(lenP*sizeof(double));
	double *gradA = calloc(lenP,sizeof(double));
	double *gradB = calloc(lenP,sizeof(double));
	double *TermA = calloc(1,sizeof(double));
	double *TermB = calloc(1,sizeof(double));
	float *evalFunc = malloc(lenY*sizeof(float));
	double TermAOld, TermBOld, funcVal, funcValStep;
	float lastStep;
	int MBox = 0; // TO REMOVE LATER
	int counter;

	omp_set_num_threads(omp_get_num_procs());	
	resetGradientFloat(gradA, gradB, TermA, TermB,lenP);
	calcGradAVXC(gradA,gradB,influence,TermA,TermB,X,XW,gridFloat,YIdx,numPointsPerBox,boxEvalPointsFloat,XToBox,numBoxes,a,b,gamma,weight,delta,n,lenY,dim,nH,MBox,evalFunc);
	sumGrad(grad,gradA,gradB,lenP);
	
	copyVector(newtonStep,grad,nH*(dim+1),1);
	// LBFGS params
	int m = (int)(nH/5) < 40 ? (int) nH/5 : 40;
	double* s_k = calloc(lenP*m,sizeof(double));
	double* y_k = calloc(lenP*m,sizeof(double));
	double* sy = calloc(m,sizeof(double));
	double* syInv = calloc(m,sizeof(double));
	double lambdaSq, step;
	int iter, numIter;
	int activeCol = 0;
	int type = 0; // 0 == 'float', 1 == 'double'
	int updateList = 0,  updateListInterval = 5;
	int switchIter = 0; // iteration in which the switch from float to double occured
	double timer; 
	//printf("%.4f, %.4f\n",*TermA, *TermB);
	// start the main iteration
	for (iter = 0; iter < 1e4; iter++) {
		timer = cpuSecond();
		// reduce hyperplanes
		if (iter > 0 && nH > 1) {
			int *activePlanes = malloc(nH*sizeof(int));
			counter = 0;
		double testSum = 0;
			//find indices of active hyperplanes
			for (i=0; i < nH; i++) {
				testSum += influence[i];
				if (influence[i] > cutoff) {
					activePlanes[counter++] = i;
				}
			}
			// if at least one hundredths is inactive
			if (counter < nH-nH/100) {
				// resize arrays
				resizeArray(&params,activePlanes,counter,nH,dim+1);
				resizeArray(&grad,activePlanes,counter,nH,dim+1);
				resizeArray(&newtonStep,activePlanes,counter,nH,dim+1);
				resizeArray(&s_k,activePlanes,counter,nH,(dim+1)*m);
				resizeArray(&y_k,activePlanes,counter,nH,(dim+1)*m);
				influence = realloc(influence,counter*sizeof(double));
				nH = counter;
				lenP = nH*(dim+1);
			}
		}
		
		lambdaSq = calcLambdaSq(grad,newtonStep,dim,nH);
		//printf("lambdaSq: %.4f\n",lambdaSq);
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
		for (i=0; i < lenP; i++) { paramsNew[i] = params[i] + newtonStep[i]; }
		unzipParams(paramsNew,aNew,bNew,dim,nH);
		// calculate gradient and objective function value
		calcGradAVXC(gradA,gradB,influence,TermA,TermB,X,XW,gridFloat,YIdx,numPointsPerBox,boxEvalPointsFloat,XToBox,numBoxes,aNew,bNew,gamma,weight,delta,n,lenY,dim,nH,MBox,evalFunc);
		sumGrad(grad,gradA,gradB,lenP);
		funcValStep = *TermA + *TermB;

		while (isnan(funcValStep) || isinf(funcValStep) || funcValStep > funcVal - step*alpha*lambdaSq) {
			if (step < 1e-9) {
				break;
			}
			step = beta*step;
			for (i=0; i < lenP; i++) {
				paramsNew[i] = params[i] + newtonStep[i]*step;
			}
			unzipParams(paramsNew,aNew,bNew,dim,nH);

			calcGradAVXC(gradA,gradB,influence,TermA,TermB,X,XW,gridFloat,YIdx,numPointsPerBox,boxEvalPointsFloat,XToBox,numBoxes,aNew,bNew,gamma,weight,delta,n,lenY,dim,nH,MBox,evalFunc);
			sumGrad(grad,gradA,gradB,lenP);
			funcValStep = *TermA + *TermB;
		}
		lastStep = funcVal - funcValStep;

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
		if (verbose > 1 && (iter < 10 || iter % 5 == 0)) {
			printf("%d: %.5f (%.4f, %.5f, %d) \t (lambdaSq: %.4e, t: %.2e, Step: %.4e) \t (Nodes per ms: %.2e) \n",iter,funcValStep,-*TermA*n,*TermB,nH,lambdaSq,step,lastStep,(lenY+n)*nH/1000/(cpuSecond() - timer));
		}
	}
	double timeB = cpuSecond();
	if (verbose > 0) {
		printf("Optimization with L-BFGS (CPU) finished: Iterations: %d, %d hyperplanes left, LogLike: %.4f, Integral: %.4e, Run time: %.2fs\n",iter,nH,(*TermA)*n,fabs(1-*TermB),timeB-timeA);
	}
	free(gradA); free(gradB); free(a); free(b); free(XDouble); free(delta); free(gridFloat); free(s_k); free(y_k); free(sy); free(syInv);
    free(grad); free(gradOld); free(newtonStep); free(paramsNew); free(params); free(evalFunc); 
}

int main() {
    const char *file = FILELOC;
    MATFile *pmat;

    /* Open matfile */
    pmat = matOpen(file, "r");
    if (pmat == NULL) {
        printf("Error opening file %s\n", file);
        return(1);
    }

   /* Matlab input variables */
    float *X = (float*)mxGetData(matGetVariable(pmat,"X"));
    float *XW = (float *) mxGetData(matGetVariable(pmat,"sW")); /* Weight vector for X */
    float *params = (float*)mxGetData(matGetVariable(pmat,"params"));
    double *box  = mxGetData(matGetVariable(pmat,"box"));
    double *ACVH  = mxGetData(matGetVariable(pmat,"ACVH"));
    double *bCVH  = mxGetData(matGetVariable(pmat,"bCVH"));

	int n = mxGetM(matGetVariable(pmat,"X")); /* number of data points */
	int dim = mxGetN(matGetVariable(pmat,"X"));
	int lenP = mxGetNumberOfElements(matGetVariable(pmat,"params")); /* number of hyperplanes */
	int lenCVH = mxGetNumberOfElements(matGetVariable(pmat,"bCVH"));
	int verbose = 1;

	double intEps = 1e-3;
	double lambdaSqEps = 1e-7;
	double cutoff = 1e-2;

	//printf("%d samples in dimension %d\n", n,dim);
	//printf("%d params, %d faces of conv(X)\n", lenP,lenCVH);

	newtonBFGSLC(X, XW, box, params, dim, lenP, n, ACVH, bCVH, lenCVH, intEps, lambdaSqEps, cutoff, verbose);
}
