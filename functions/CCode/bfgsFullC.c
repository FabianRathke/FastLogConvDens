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
extern void calcGradAVXC(double* gradA, double* gradB, double* influence, double* TermA, double* TermB, float* X, float* XW, float* grid, unsigned short int* YIdx, int *numPointsPerBox, float* boxEvalPoints, unsigned short int *XToBox, int numBoxes, double* a, double* b, float gamma, float weight, float* delta, int N, int M, int dim, int nH, int MBox);
extern void calcGradC(double* gradA, double* gradB, double* influence, double* TermA, double* TermB, double* X, double* XW, double* grid, unsigned short int* YIdx, int *numPointsPerBox, double* boxEvalPoints, unsigned short int *XToBox, int numBoxes, double* a, double* b, double gamma, double weight, double* delta, int N, int M, int dim, int nH, int MBox);
extern void preCondGradAVXC(int** elementList, int** elementListSize, int* numEntries, int* maxElement, int* idxEntries, float* X, float* grid, unsigned short int* YIdx, int *numPointsPerBox, float* boxEvalPoints, int numBoxes, double* a, double* aTrans, double* b, float gamma, float weight, float* delta, int N, int M, int dim, int nH);
extern void calcGradFastFloatC(int* numEntries, int* elementList, int* maxElement, int* idxEntries, double* grad, double* influence, double* TermA, double* TermB, float* X, float* XW, float* grid, unsigned short int* YIdx, double* a, double* b, float gamma, float weight, float* delta, int N, int M, int dim, int nH);
extern void calcGradFastC(int* numEntries, int* elementList, int* maxElement, int* idxEntries, double* grad, double* influence, double* TermA, double* TermB, double* X, double* XW, double* grid, unsigned short int* YIdx, double* a, double* b, double gamma, double weight, double* delta, int N, int M, int dim, int nH);

// TODO: Don't copy, only set points for *b and *a (of transpose == 0)
void unzipParams(double *params, double *a, double *b, int dim, int nH, int transpose) {
	int i,j;
	if (transpose==1) {
		// transpose operation
		for (i=0; i < dim; i++) {
		   for (j=0; j < nH; j++) {
			   a[j*dim + i] = params[j + i*nH];
		   }
		}
	} else {
		for (i=0; i < dim*nH; i++) {
			a[i] = params[i];
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


void sumVec(double *A, double *B, double *C, int n) {
	for (int i=0; i < n; i++) {
		A[i] = B[i] + C[i];
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

void resetGradient(double* gradA, double* gradB, double* TermA, double* TermB, int lenP) {
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

void resizeCNSarray(double **a, int c, int c_, int activeCol, int lenP, int m) {
	if (c_!=-1) {
		memcpy(*a+c_*lenP,*a,(activeCol+1)*lenP*sizeof(double));
    	memcpy(*a,*a+c*lenP,c_*lenP*sizeof(double));
	} else {
		memcpy(*a,*a+c*lenP,lenP*m*sizeof(double));
	}
	*a = realloc(*a,m*lenP*sizeof(double));
}

void cumsum(int* numEntriesCumSum, int* numEntries, int n) {
	numEntriesCumSum[0] = 0;
	numEntriesCumSum[1] = numEntries[0];
	for (int i = 1; i < n; i++) {
		numEntriesCumSum[i+1] = numEntries[i]+numEntriesCumSum[i];
	}
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
void newtonBFGSLC(double* X,  double* XW, double* box, double* params_, int dim, int *lenP, int n, double* ACVH, double* bCVH, int lenCVH, double intEps, double lambdaSqEps, double cutoff, int verbose) {

	int i;
	double timeA = cpuSecond();
	
	// number of hyperplanes
	int nH  = (int) *lenP/(dim+1);

	// create the integration grid
    int lenY, numBoxes = 0;
	int *numPointsPerBox; unsigned short int *YIdx, *XToBox; double *boxEvalPoints; 

	// obtain grid density params
	int NGrid, MGrid;
    double weight = 0; 
    double *grid = NULL;
    setGridDensity(box,dim,0,&NGrid,&MGrid,&grid,&weight);
	
	float *delta = malloc(dim*sizeof(float));
	double *deltaD = malloc(dim*sizeof(double));
	for (i=0; i < dim; i++) {
		delta[i] = grid[NGrid*MGrid*i+1] - grid[NGrid*MGrid*i];
		deltaD[i] = grid[NGrid*MGrid*i+1] - grid[NGrid*MGrid*i];
	}
	
	float *XF = malloc(n*dim*sizeof(float)); 	for (i=0; i < n*dim; i++) { XF[i] = X[i]; }
	float *XWF = malloc(n*sizeof(float)); for (i=0; i < n; i++) { XWF[i] = XW[i]; }
	double *params = malloc(*lenP*sizeof(double)); for (i=0; i < *lenP; i++) {params[i] = params_[i]; }


	//omp_set_num_threads(omp_get_num_procs());
	omp_set_num_threads(2);
	//printf("%d, %d\n",omp_get_num_procs(), omp_get_max_threads());
 
	//printf("Obtain grid for N = %d and M = %d\n",NGrid,MGrid);
	makeGridC(X,&YIdx,&XToBox,&numPointsPerBox,&boxEvalPoints,ACVH,bCVH,box,&lenY,&numBoxes,dim,lenCVH,NGrid,MGrid,n);
	//printf("Obtained grid with %d points and %d boxes\n",lenY,numBoxes);

	float *boxEvalPointsFloat = malloc(numBoxes*dim*3*sizeof(float));
	for (i=0; i < numBoxes*dim*3; i++) { boxEvalPointsFloat[i] = (float) boxEvalPoints[i]; }
	// only the first entry in each dimension is required
	float *gridFloat = malloc(dim*sizeof(float));
	double *gridDouble = malloc(dim*sizeof(double));
	for (i=0; i < dim; i++) {
		gridFloat[i] = grid[i*NGrid*MGrid];
		gridDouble[i] = (double) grid[i*NGrid*MGrid];
	}
	// two points for a and b: slope and bias of hyperplanes
	double *a = malloc(nH*dim*sizeof(double));
	double *b = malloc(nH*sizeof(double));
	double *aTrans = NULL; 
	unzipParams(params,a,b,dim,nH,1);

	double *influence = malloc(nH*sizeof(double));
	double alpha = 1e-4, beta = 0.1;
	double gamma = 1000, timer;

	double *grad = malloc(*lenP*sizeof(double)), *gradCheck =  NULL;
	double *gradOld = malloc(*lenP*sizeof(double));
	double *newtonStep = malloc(*lenP*sizeof(double));
	double *paramsNew = malloc(*lenP*sizeof(double));
	double *gradA = calloc(*lenP,sizeof(double));
	double *gradB = calloc(*lenP,sizeof(double));
	double *TermA = calloc(1,sizeof(double));
	double *TermB = calloc(1,sizeof(double));
	double TermAOld, TermBOld, funcVal, funcValStep, lastStep;
	int counterActive, counterInactive;

	resetGradient(gradA, gradB, TermA, TermB,*lenP);
	calcGradAVXC(gradA,gradB,influence,TermA,TermB,XF,XWF,gridFloat,YIdx,numPointsPerBox,boxEvalPointsFloat,XToBox,numBoxes,a,b,gamma,weight,delta,n,lenY,dim,nH,MGrid);
	sumVec(grad,gradA,gradB,*lenP);

	copyVector(newtonStep,grad,nH*(dim+1),1);
	// LBFGS params
	int m = (int)(nH/5) < 40 ? (int) nH/5 : 40;
	double *s_k = calloc(*lenP*m,sizeof(double));
	double *y_k = calloc(*lenP*m,sizeof(double));
	double *sy = calloc(m,sizeof(double));
	double *syInv = calloc(m,sizeof(double));
	double lambdaSq, step;
	int iter, numIter;
	int activeCol = 0;
	int type = 0; // 0 == 'float', 1 == 'double'
	int mode = 0; // 0 == 'normal', 1 == 'fast' - the fast mode keeps a list of active hyperplanes for each sample and grid points which gets updated every updateListInterval interations
	int updateList = 0,  updateListInterval = 5;
	int switchIter = -40; // iteration in which the switch from float to double occured
	int maxIter = 1e4;
	int *nHHist = malloc(maxIter*sizeof(int)), *activePlanes = NULL, *inactivePlanes = NULL;
	int *elementListSize = NULL, *elementList = NULL, *numEntries = NULL, *maxElement=NULL, *idxEntries=NULL, *numEntriesCumSum = NULL;
	// start the main iteration
	for (iter = 0; iter < maxIter; iter++) {
		nHHist[iter] = nH;
		timer = cpuSecond();
		updateList--;
		// reduce hyperplanes
		if (iter > 0 && nH > 1) {
			free(activePlanes); free(inactivePlanes);
			activePlanes = malloc(nH*sizeof(int)); inactivePlanes = malloc(nH*sizeof(int));
			counterActive = 0; 	counterInactive = 0;

			//find indices of active hyperplanes
			for (i=0; i < nH; i++) {
				if (influence[i] > cutoff) {
					activePlanes[counterActive++] = i;
				} else {
					inactivePlanes[counterInactive++] = i;
				}
			}

			if (counterInactive > counterActive) {
				while (counterInactive > counterActive) {
					activePlanes[counterActive++] = inactivePlanes[--counterInactive];
				}
			}

			// if at least one hundredths is inactive remove these hyperplanes
			if (counterActive < nH-nH/100) {
				// resize arrays
				resizeArray(&params,activePlanes,counterActive,nH,dim+1);
				resizeArray(&grad,activePlanes,counterActive,nH,dim+1);
				resizeArray(&newtonStep,activePlanes,counterActive,nH,dim+1);
				resizeArray(&s_k,activePlanes,counterActive,nH,(dim+1)*m);
				resizeArray(&y_k,activePlanes,counterActive,nH,(dim+1)*m);
				influence = realloc(influence,counterActive*sizeof(double));
				nH = counterActive;
				*lenP = nH*(dim+1);

				if (mode==1) { // update list of active hyperplanes for all samples/grid points 
					unzipParams(params,aTrans,b,dim,nH,1);
					unzipParams(params,a,b,dim,nH,0);
					preCondGradAVXC(&elementList,&elementListSize,numEntries,maxElement,idxEntries,XF,gridFloat,YIdx,numPointsPerBox,boxEvalPointsFloat,numBoxes,a,aTrans,b,gamma,weight,delta,n,lenY,dim,nH); 
					cumsum(numEntriesCumSum,numEntries,n+lenY+1);

					if (counterActive < nH-5*nH/100) {
						updateListInterval = (int) updateListInterval/2;
					}
					updateList = updateListInterval;
				}

				// adapt m to reduced problem size
				if (m > (int) (*lenP/10)) {
					int mOld = m;
					m = (int) (*lenP/2) <  (int) (m/2) ? (int) (*lenP/2) : (int) (m/2);
					//printf("entered: %d, %d, %d\n",mOld,m,activeCol);
					int c, c_;
					if (activeCol >= m-1) {
						c = activeCol-m+1;
						c_ = -1;
					} else {
						c = activeCol+1+mOld-m;
						c_ = m-activeCol-1;
					}
					resizeCNSarray(&sy,c,c_,activeCol,1,m);
					resizeCNSarray(&syInv,c,c_,activeCol,1,m);
					resizeCNSarray(&s_k,c,c_,activeCol,*lenP,m);
					resizeCNSarray(&y_k,c,c_,activeCol,*lenP,m);
			
					if (iter >= m) {
						activeCol = m-1;
					}
				}
			}
		}

		// switch to sparse approximative mode
		if (iter >= 25 && ((double) nHHist[iter-25] - nHHist[iter])/(double) nHHist[iter] < 0.05 && mode == 0 && nH > 500 && gamma >= 100) {
			mode = 1;
			if (verbose > 1) {
				printf("Changed	mode\n");
			}
			updateList = updateListInterval;
			
			numEntries = malloc((n+lenY)*sizeof(int));
			numEntriesCumSum = malloc((n+lenY+1)*sizeof(int));
			maxElement = malloc((n+lenY)*sizeof(int));
			idxEntries = malloc(lenY*sizeof(int));
			aTrans = malloc(nH*dim*sizeof(double));
			// we require both the transposed and the normal variant of a
			unzipParams(params,aTrans,b,dim,nH,1);
			unzipParams(params,a,b,dim,nH,0);
			preCondGradAVXC(&elementList,&elementListSize,numEntries,maxElement,idxEntries,XF,gridFloat,YIdx,numPointsPerBox,boxEvalPointsFloat,numBoxes,a,aTrans,b,gamma,weight,delta,n,lenY,dim,nH); 
			cumsum(numEntriesCumSum,numEntries,n+lenY+1);
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
		// add newtonStep to params vector
		sumVec(paramsNew,params,newtonStep,*lenP);
		// calculate gradient and objective function value
		if (mode == 0) { // normal mode
			if (type == 0) { // single
				unzipParams(paramsNew,a,b,dim,nH,1);
				calcGradAVXC(gradA,gradB,influence,TermA,TermB,XF,XWF,gridFloat,YIdx,numPointsPerBox,boxEvalPointsFloat,XToBox,numBoxes,a,b,gamma,weight,delta,n,lenY,dim,nH,MGrid);
			} else { // double
				unzipParams(paramsNew,a,b,dim,nH,0);
				calcGradC(gradA,gradB,influence,TermA,TermB,X,XW,gridDouble,YIdx,numPointsPerBox,boxEvalPoints,XToBox,numBoxes,a,b,gamma,weight,deltaD,n,lenY,dim,nH,MGrid);
			}
			sumVec(grad,gradA,gradB,*lenP);
		} else { // aproximative mode
			if (type == 0) { // single
				unzipParams(paramsNew,a,b,dim,nH,1);
				calcGradFastFloatC(numEntriesCumSum,elementList,maxElement,idxEntries,grad,influence,TermA,TermB,XF,XWF,gridFloat,YIdx,a,b,gamma,weight,delta,n,lenY,dim,nH);
			} else { // double
				unzipParams(paramsNew,a,b,dim,nH,0);
				calcGradFastC(numEntriesCumSum,elementList,maxElement,idxEntries,grad,influence,TermA,TermB,X,XW,gridDouble,YIdx,a,b,gamma,weight,deltaD,n,lenY,dim,nH);
			}
			// update elementlist
			if (updateList < 0) {
			   	//printf("Update elementList\n");
				unzipParams(paramsNew,aTrans,b,dim,nH,1);
            	unzipParams(paramsNew,a,b,dim,nH,0);
				preCondGradAVXC(&elementList,&elementListSize,numEntries,maxElement,idxEntries,XF,gridFloat,YIdx,numPointsPerBox,boxEvalPointsFloat,numBoxes,a,aTrans,b,gamma,weight,delta,n,lenY,dim,nH); 
				cumsum(numEntriesCumSum,numEntries,n+lenY+1);

				// check whether the control interval has to be reduced
				gradCheck = malloc(*lenP*sizeof(double));
				memcpy(gradCheck,grad,*lenP*sizeof(double));
				if (type == 0) {
					unzipParams(paramsNew,a,b,dim,nH,1);
					calcGradAVXC(gradA,gradB,influence,TermA,TermB,XF,XWF,gridFloat,YIdx,numPointsPerBox,boxEvalPointsFloat,XToBox,numBoxes,a,b,gamma,weight,delta,n,lenY,dim,nH,MGrid);
				} else {
					unzipParams(paramsNew,a,b,dim,nH,0);
					calcGradC(gradA,gradB,influence,TermA,TermB,X,XW,gridDouble,YIdx,numPointsPerBox,boxEvalPoints,XToBox,numBoxes,a,b,gamma,weight,deltaD,n,lenY,dim,nH,MGrid);
				}
				double normGrad = 0;
				for (i=0; i < *lenP; i++) { normGrad += (grad[i]-gradCheck[i])*(grad[i]-gradCheck[i]); }
				if (sqrt(normGrad) < 1e-5) {
					updateListInterval = updateListInterval*2 > 100 ? 100 : updateListInterval*2;
				} else {
					updateListInterval = updateListInterval/2;
				}
				updateList = updateListInterval;
			}
		}
		funcValStep = *TermA + *TermB;

		while (isnan(funcValStep) || isinf(funcValStep) || funcValStep > funcVal - step*alpha*lambdaSq) {
			if (step < 1e-9) {
				break;
			}
			step = beta*step;
			for (i=0; i < *lenP; i++) { paramsNew[i] = params[i] + newtonStep[i]*step; }
			if (mode == 0) {
				if (type == 0) {
					unzipParams(paramsNew,a,b,dim,nH,1);
					calcGradAVXC(gradA,gradB,influence,TermA,TermB,XF,XWF,gridFloat,YIdx,numPointsPerBox,boxEvalPointsFloat,XToBox,numBoxes,a,b,gamma,weight,delta,n,lenY,dim,nH,MGrid);
				} else {
					unzipParams(paramsNew,a,b,dim,nH,0);
					calcGradC(gradA,gradB,influence,TermA,TermB,X,XW,gridDouble,YIdx,numPointsPerBox,boxEvalPoints,XToBox,numBoxes,a,b,gamma,weight,deltaD,n,lenY,dim,nH,MGrid);
				}
				sumVec(grad,gradA,gradB,*lenP);
			} else {
				if (type == 0) {
					unzipParams(paramsNew,a,b,dim,nH,1);
					calcGradFastFloatC(numEntriesCumSum,elementList,maxElement,idxEntries,grad,influence,TermA,TermB,XF,XWF,gridFloat,YIdx,a,b,gamma,weight,delta,n,lenY,dim,nH);
				} else {
					unzipParams(paramsNew,a,b,dim,nH,0);
					calcGradFastC(numEntriesCumSum,elementList,maxElement,idxEntries,grad,influence,TermA,TermB,X,XW,gridDouble,YIdx,a,b,gamma,weight,deltaD,n,lenY,dim,nH);
				}
			}

			funcValStep = *TermA + *TermB;
		}
		lastStep = funcVal - funcValStep;

		for (i=0; i < *lenP; i++) { params[i] = paramsNew[i]; }
	
		// convert to double if increased precision is required
		if (lastStep <= 0 && type == 0) {
			type = 1;
			switchIter = iter;
			if (verbose > 1) {
				printf("Switch to double\n");
			}
		}
	
		if (fabs(1-*TermB) < intEps && lastStep < lambdaSqEps && iter > 10 && iter - switchIter > 50) {
			break;
		}
	
		// min([m,iter,length(params)]) --> C indexing of iter is one less than matlab --> +1
		numIter = m < iter+1 ? m : iter+1;
		numIter = *lenP < numIter ? *lenP : numIter;
		CNS(s_k,y_k,sy,syInv,step,grad,gradOld,newtonStep,numIter,activeCol,*lenP,m);
		activeCol++; 
    	if (activeCol >= m) {
        	activeCol = 0;
		}
		double timeB = cpuSecond()-timer;
		if (verbose > 1 && (iter < 10 || iter % 5 == 0)) {
			printf("%d: %.5f (%.4f, %.5f, %d) \t (lambdaSq: %.4e, t: %.0e, Step: %.4e) \t (Nodes per ms: %.2e)  %d \n",iter,funcValStep,-*TermA*n,*TermB,nH,lambdaSq,step,lastStep,(lenY+n)*nH/1000/(timeB), updateListInterval);
		}
	}
	double timeB = cpuSecond();
	if (verbose > 0) {
		printf("Optimization with L-BFGS (CPU) finished: %d Iterations, %d hyperplanes left, LogLike: %.4f, Integral: %.4e, Run time: %.2fs\n",iter,nH,(*TermA)*n,fabs(1-*TermB),timeB-timeA);
	}
	memcpy(params_,params,*lenP*sizeof(double));

	free(delta); free(deltaD); free(XF); free(XWF); free(params); free(boxEvalPointsFloat); free(gridFloat); free(gridDouble); free(a); free(b); free(aTrans); free(influence);
	free(grad); free(gradOld); free(gradA); free(gradB); free(newtonStep); free(paramsNew); free(nHHist); free(activePlanes); free(inactivePlanes); free(gradCheck);
	free(numEntries); free(numEntriesCumSum); free(idxEntries); free(maxElement); free(s_k); free(y_k); free(sy); free(syInv);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
   /* Matlab input variables */
    double *X = (double*)mxGetData(prhs[0]);
    double *XW = (double *) mxGetData(prhs[1]); /* Weight vector for X */
    double *params = (double*)mxGetData(prhs[2]);
    double *box  = mxGetData(prhs[3]);
    double *ACVH  = mxGetData(prhs[4]);
    double *bCVH  = mxGetData(prhs[5]);
	int verbose = (int) mxGetScalar(prhs[6]);

	int n = mxGetM(prhs[0]); /* number of data points */
	int dim = mxGetN(prhs[0]);
	int lenP = mxGetNumberOfElements(prhs[2]); /* number of hyperplanes */
	int lenCVH = mxGetNumberOfElements(prhs[5]);

	double intEps = 1e-3;
	double lambdaSqEps = 1e-7;
	double cutoff = 1e-2;

	printf("%d samples in %dD, %d initial hyperplanes\n", n,dim,lenP/(dim+1));

	newtonBFGSLC(X, XW, box, params, dim, &lenP, n, ACVH, bCVH, lenCVH, intEps, lambdaSqEps, cutoff, verbose);
	
    plhs[0] = mxCreateNumericMatrix(lenP,1,mxDOUBLE_CLASS,mxREAL);
	memcpy(mxGetPr(plhs[0]),params,lenP*sizeof(double));
}
