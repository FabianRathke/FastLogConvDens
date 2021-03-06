#include <mex.h>
#include <mat.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <omp.h>

extern double cpuSecond();
extern void makeGridC(double *X, unsigned short int **YIdx, unsigned short int **XToBox, int **numPointsPerBox, double **boxEvalPoints, double *ACVH, double *bCVH, double *box, int *lenY, int *numBoxes, int dim, int lenCVH, int NX, double ratio, int minGridSize, int *NGrid, int *MGrid, double **grid, double *weight);
extern void CNS(double* s_k, double *y_k, double *sy, double *syInv, double step, double *grad, double *gradOld, double *newtonStep, int numIter, int activeCol, int nH, int m);
extern void calcGradAVXC(double* gradA, double* gradB, double* influence, double* TermA, double* TermB, float* X, float* XW, float* grid, unsigned short int* YIdx, int *numPointsPerBox, float* boxEvalPoints, unsigned short int *XToBox, int numBoxes, double* a, double* b, float gamma, float weight, float* delta, int N, int M, int dim, int nH);
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
		memcpy(dest,source,n*sizeof(double));
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


typedef struct {
	int id; 
	int XToBox;
} point;

int compare (const void * a, const void * b)
{
  point *pointA = (point *)a;
  point *pointB = (point *)b;

  return ( pointA->XToBox - pointB->XToBox );
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
void newtonBFGSLC(double *X_,  double *XW_, double *box, double *params_, double *paramsB, int *lenP, int lenPB, int dim, int n, double *ACVH, double *bCVH, int lenCVH, double intEps, double lambdaSqEps, double cutoff, int verbose, double **grid_, unsigned short int **YIdx_, int *lenY_, int *NGrid_, int *MGrid_, double *weight_, double gamma, double ratio, int minGridSize) {

	//omp_set_num_threads(omp_get_max_threads());
	omp_set_num_threads(4);
	int numThreads = 0;
	if (verbose > 1) {
		#pragma omp parallel
		{
			numThreads = omp_get_num_threads();
		}
		printf("%d Threads used\n", numThreads);
	}

	int i;
	double timeA = cpuSecond();
	
	// create the integration grid
    int lenY, numBoxes = 0;
	int *numPointsPerBox; unsigned short int *YIdx, *XToBox; double *boxEvalPoints; 
	
	// obtain grid density params
	int NGrid, MGrid;
    double weight = 0; 
	double *grid = NULL;
    //setGridDensity(box,dim,0,&NGrid,&MGrid,&grid,&weight,ratio, minGridSize);

	//makeGridC(X_,&YIdx,&XToBox,&numPointsPerBox,&boxEvalPoints,ACVH,bCVH,box,&lenY,&numBoxes,dim,lenCVH,NGrid,MGrid,n);
	makeGridC(X_,&YIdx,&XToBox,&numPointsPerBox,&boxEvalPoints,ACVH,bCVH,box,&lenY,&numBoxes,dim,lenCVH,n,ratio,minGridSize,&NGrid, &MGrid,&grid,&weight);
	if (verbose > 1) {
		printf("Obtain grid for N = %d and M = %d\n",NGrid,MGrid);
	}
	if (verbose > 1) {
		printf("Obtained grid with %d points and %d boxes\n",lenY,numBoxes);
	}

	// copy values --> return values to matlab in mex file
	*lenY_ = lenY;
	*MGrid_ = MGrid; *NGrid_ = NGrid;
	*YIdx_ = YIdx; 
	*grid_ = grid;
	*weight_ = weight;
	
	
	point list[n];
	for (i=0; i < n; i++) {
		//printf("%d: %d (%.4f, %.4f)\n",i,XToBox[i], X_[i], X_[i + n]);
		list[i].id = i; 
		list[i].XToBox = XToBox[i];
	}

	qsort(list,n,sizeof(point),compare);
	//int *B = malloc(n*sizeof(int));
	double *X = malloc(n*dim*sizeof(double)); // sort X according to the boxes it is in
	double *XW = malloc(n*sizeof(double)); // sort XW according to the boxes it is in
	for (i=0; i < n; i++) {
		for (int k=0; k < dim; k++) {
			X[i+k*n] = X_[list[i].id + k*n];
		}
		XW[i] = XW_[list[i].id];
		XToBox[i] = list[i].XToBox;
	}
	
	float *boxEvalPointsFloat = malloc(numBoxes*dim*3*sizeof(float));
	for (i=0; i < numBoxes*dim*3; i++) { boxEvalPointsFloat[i] = (float) boxEvalPoints[i]; }
	// only the first entry in each dimension is required
	float *gridFloat = malloc(dim*sizeof(float));
	double *gridDouble = malloc(dim*sizeof(double));
	for (i=0; i < dim; i++) {
		gridFloat[i] = grid[i*NGrid*MGrid];
		gridDouble[i] = (double) grid[i*NGrid*MGrid];
	}

	float *delta = malloc(dim*sizeof(float));
	double *deltaD = malloc(dim*sizeof(double));
	for (i=0; i < dim; i++) {
		delta[i] = grid[NGrid*MGrid*i+1] - grid[NGrid*MGrid*i];
		deltaD[i] = grid[NGrid*MGrid*i+1] - grid[NGrid*MGrid*i];
	}
	
	float *XF = malloc(n*dim*sizeof(float)); 	for (i=0; i < n*dim; i++) { XF[i] = X[i]; }
	float *XWF = malloc(n*sizeof(float)); for (i=0; i < n; i++) { XWF[i] = XW[i]; }
	double timer;

	//choose between initializations
	int nH  = (int) *lenP/(dim+1);
	double *gradA = calloc(*lenP,sizeof(double));
	double *gradB = calloc(*lenP,sizeof(double));
	double *TermA = calloc(1,sizeof(double));
	double *TermB = calloc(1,sizeof(double));
	double *a = malloc(nH*dim*sizeof(double));
	double *b = malloc(nH*sizeof(double));
	double *influence = malloc(nH*sizeof(double));

	unzipParams(params_,a,b,dim,nH,1);
	calcGradAVXC(gradA,gradB,influence,TermA,TermB,XF,XWF,gridFloat,YIdx,numPointsPerBox,boxEvalPointsFloat,XToBox,numBoxes,a,b,gamma,weight,delta,n,lenY,dim,nH);
	double initA = *TermA + *TermB;
	//printf("TermA: %.4f, TermB: %.4f\n",*TermA, *TermB);
		
	double *params = NULL;
	if (lenPB > 0) {
		// paramsB
		int nHB = (int) lenPB/(dim+1);
		double *gradAB = calloc(lenPB,sizeof(double));
		double *gradBB = calloc(lenPB,sizeof(double));
		double *TermAB = calloc(1,sizeof(double));
		double *TermBB = calloc(1,sizeof(double));
		double *aB = malloc(nHB*dim*sizeof(double));
		double *bB = malloc(nHB*sizeof(double));
		double *influenceB = malloc(nHB*sizeof(double));

		unzipParams(paramsB,aB,bB,dim,nHB,1);
		calcGradAVXC(gradAB,gradBB,influenceB,TermAB,TermBB,XF,XWF,gridFloat,YIdx,numPointsPerBox,boxEvalPointsFloat,XToBox,numBoxes,aB,bB,gamma,weight,delta,n,lenY,dim,nHB);
		double initB = *TermAB + *TermBB;
		printf("TermA: %.4f, TermB: %.4f\n",initA, initB);

		if (initA < initB) {
			if (verbose > 1) {
				printf("Choose log-concave density with gamma = 1 for initialization\n");
			}
			free(gradAB); free(gradBB); free(TermAB); free(TermBB); free(aB); free(bB); free(influenceB);
			params = malloc(*lenP*sizeof(double)); memcpy(params,params_,*lenP*sizeof(double)); 
		} else {
			if (verbose > 1) {
				printf("Choose kernel density for initialization\n");
			}
			free(gradA); free(gradB); free(TermA); free(TermB); free(a); free(b); free(influence);
			gradA = gradAB; gradB = gradBB; TermA = TermAB; TermB = TermBB; a = aB; b = bB; influence = influenceB;
			*lenP = lenPB; nH = nHB;
			params = malloc(*lenP*sizeof(double)); memcpy(params,paramsB,*lenP*sizeof(double)); 
		}
	} else {
		params = malloc(*lenP*sizeof(double)); memcpy(params,params_,*lenP*sizeof(double)); 
	}

	if (verbose > 1) {
		printf("******* Run optimization on %d grid points for %d hyperplanes ***********\n",lenY,*lenP/(dim+1));
	}

	// two points for a and b: slope and bias of hyperplanes
	double *aTrans = NULL; 
	double alpha = 1e-4, beta = 0.1;
	
	double *grad = malloc(*lenP*sizeof(double)), *gradCheck =  NULL;
	double *gradOld = malloc(*lenP*sizeof(double));
	double *newtonStep = malloc(*lenP*sizeof(double));
	double *paramsNew = malloc(*lenP*sizeof(double));
	double TermAOld, TermBOld, funcVal, funcValStep, lastStep;
	int counterActive, counterInactive;

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
	int maxIter = 1e5;
	int maxUpdateInterval = 50;
	double lastStepSave = 100;
	//maxIter = 1;
	int *nHHist = malloc(maxIter*sizeof(int)), *activePlanes = NULL, *inactivePlanes = NULL;
	int *elementListSize = NULL, *elementList = NULL, *numEntries = NULL, *maxElement=NULL, *idxEntries=NULL, *numEntriesCumSum = NULL;
	// start the main iteration
	for (iter = 0; iter < maxIter; iter++) {
		nHHist[iter] = nH;
		timer = cpuSecond();
		updateList--;
		// reduce hyperplanes
		if (iter > 0 && nH > 1 && iter > switchIter + 30) {
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

			// remove superfluous inactive hyperplanes 
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
					cumsum(numEntriesCumSum,numEntries,n+lenY);

					if (counterActive < nH-5*nH/100) {
						updateListInterval = (int) updateListInterval/2;
					}
					updateList = updateListInterval;
				}

				// adapt m to reduced problem size
				if (m > (int) (*lenP/5) && m > 1) {
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
			cumsum(numEntriesCumSum,numEntries,n+lenY);
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
				calcGradAVXC(gradA,gradB,influence,TermA,TermB,XF,XWF,gridFloat,YIdx,numPointsPerBox,boxEvalPointsFloat,XToBox,numBoxes,a,b,gamma,weight,delta,n,lenY,dim,nH);
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
				cumsum(numEntriesCumSum,numEntries,n+lenY);

				// check whether the control interval has to be reduced
				gradCheck = malloc(*lenP*sizeof(double));
				memcpy(gradCheck,grad,*lenP*sizeof(double));
				if (type == 0) {
					unzipParams(paramsNew,a,b,dim,nH,1);
					calcGradAVXC(gradA,gradB,influence,TermA,TermB,XF,XWF,gridFloat,YIdx,numPointsPerBox,boxEvalPointsFloat,XToBox,numBoxes,a,b,gamma,weight,delta,n,lenY,dim,nH);
				} else {
					unzipParams(paramsNew,a,b,dim,nH,0);
					calcGradC(gradA,gradB,influence,TermA,TermB,X,XW,gridDouble,YIdx,numPointsPerBox,boxEvalPoints,XToBox,numBoxes,a,b,gamma,weight,deltaD,n,lenY,dim,nH,MGrid);
				}
				double normGrad = 0;
				for (i=0; i < *lenP; i++) { normGrad += (grad[i]-gradCheck[i])*(grad[i]-gradCheck[i]); }
				if (sqrt(normGrad) < 1e-5) {
					updateListInterval = updateListInterval*2 > maxUpdateInterval ? maxUpdateInterval : updateListInterval*2;
				} else {
					updateListInterval = updateListInterval/2;
				}
				updateList = updateListInterval;
			}
		}
		funcValStep = *TermA + *TermB;
		
		// Armijo backtracking
		while (isnan(funcValStep) || isinf(funcValStep) || funcValStep > funcVal - step*alpha*lambdaSq) {
			if (step < 1e-9) {
				break;
			}
			step = beta*step;
			for (i=0; i < *lenP; i++) { paramsNew[i] = params[i] + newtonStep[i]*step; }
			if (mode == 0) {
				if (type == 0) {
					unzipParams(paramsNew,a,b,dim,nH,1);
					calcGradAVXC(gradA,gradB,influence,TermA,TermB,XF,XWF,gridFloat,YIdx,numPointsPerBox,boxEvalPointsFloat,XToBox,numBoxes,a,b,gamma,weight,delta,n,lenY,dim,nH);
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
		if (iter > 1) {
			lastStepSave = lastStep;
		}
		lastStep = funcVal - funcValStep;

		memcpy(params,paramsNew,*lenP*sizeof(double));
	
		// convert to double if increased precision is required
		if (lastStep <= 0 && type == 0) {
			type = 1;
			switchIter = iter;
			if (verbose > 1) {
				printf("Switch to double\n");
			}
		}

		if (lastStep <= 0 && type  == 1) {
			updateList = -1;
			updateListInterval = 5;
		}

	
		if (fabs(1-*TermB) < intEps && lastStep < lambdaSqEps && lastStepSave < lambdaSqEps && iter > 10 && iter - switchIter > 50) {
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
			printf("%d: %.5f (%.4f, %.5f, %d) \t (lambdaSq: %.4e, t: %.0e, Step: %.4e) \t (Nodes per ms: %.2e)  %d \n",iter,funcValStep,-*TermA*n,*TermB,nH,lambdaSq,step,lastStep,(lenY+n)/1000/timeB*nH, updateListInterval);
		}
	}
	double timeB = cpuSecond();
	if (verbose > 0) {
		printf("Optimization with L-BFGS (CPU) finished: %d Iterations, %d hyperplanes remaining, LogLike: %.4f, Integral: %.4e, Run time: %.2fs\n",iter,nH,(*TermA)*n,fabs(1-*TermB),timeB-timeA);
	}
	memcpy(params_,params,*lenP*sizeof(double));

	free(delta); free(deltaD); free(XF); free(XWF); free(params); free(boxEvalPointsFloat); free(gridFloat); free(gridDouble); free(a); free(b); free(aTrans); free(influence);
	free(grad); free(gradOld); free(gradA); free(gradB); free(newtonStep); free(paramsNew); free(nHHist); free(activePlanes); free(inactivePlanes); free(gradCheck);
	free(s_k); free(y_k); free(sy); free(syInv);
	free(X); free(XW); free(TermA); free(TermB);
	// free grid variables
	free(numPointsPerBox); free(XToBox); free(boxEvalPoints); // free(YIdx); free(grid);
	// free preconditioner variables
	free(numEntries); free(numEntriesCumSum); free(idxEntries); free(maxElement); free(elementListSize); free(elementList);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
   /* Matlab input variables */
    double *X = (double*)mxGetData(prhs[0]);
    double *XW = (double *) mxGetData(prhs[1]); /* Weight vector for X */
    double *paramsA = (double*)mxGetData(prhs[2]);
    double *paramsB = (double*)mxGetData(prhs[3]);
    double *box  = mxGetData(prhs[4]);
    double *ACVH  = mxGetData(prhs[5]);
    double *bCVH  = mxGetData(prhs[6]);
	int verbose = (int) mxGetScalar(prhs[7]);
	double intEps = mxGetScalar(prhs[8]);
	double lambdaSqEps = mxGetScalar(prhs[9]);
	double cutoff = mxGetScalar(prhs[10]);
	double gamma = mxGetScalar(prhs[11]);
	double ratio = mxGetScalar(prhs[12]);
	int minGridSize = (int) mxGetScalar(prhs[13]);

	int n = mxGetM(prhs[0]); /* number of data points */
	int dim = mxGetN(prhs[0]);
	int lenPA = mxGetNumberOfElements(prhs[2]); /* number of hyperplanes */
	int lenPB = mxGetNumberOfElements(prhs[3]); /* number of hyperplanes */
	int lenCVH = mxGetNumberOfElements(prhs[6]);


	unsigned short int *YIdx;
	double *grid;
	int lenY, NGrid, MGrid;
	double weight;
	newtonBFGSLC(X, XW, box, paramsA, paramsB, &lenPA, lenPB, dim, n, ACVH, bCVH, lenCVH, intEps, lambdaSqEps, cutoff, verbose, &grid, &YIdx, &lenY, &NGrid, &MGrid, &weight, gamma, ratio, minGridSize);
	
    plhs[0] = mxCreateNumericMatrix(lenPA,1,mxDOUBLE_CLASS,mxREAL);
	memcpy(mxGetPr(plhs[0]),paramsA,lenPA*sizeof(double));

	plhs[1] = mxCreateNumericMatrix(dim,lenY,mxUINT16_CLASS,mxREAL);
	memcpy(mxGetPr(plhs[1]),YIdx,lenY*dim*sizeof(unsigned short int));

	plhs[2] = mxCreateNumericMatrix(NGrid*MGrid,dim,mxDOUBLE_CLASS,mxREAL);
	memcpy(mxGetPr(plhs[2]),grid,NGrid*MGrid*dim*sizeof(double));

	plhs[3] = mxCreateNumericMatrix(1,1,mxDOUBLE_CLASS, mxREAL);
	memcpy(mxGetPr(plhs[3]),&weight,sizeof(double));
}
