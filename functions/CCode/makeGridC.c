#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <limits.h>

void makeGridMidpoint(double* box, double* grid, int N, int dim) {
	int i,j;
	double interval;
    for (i=0; i < dim; i++) {   
        interval = (box[i+dim]-box[i])/N;
        for (j=0; j < N; j++) {
            grid[j + i*N] = box[i] + interval*(j+0.5);
        }
    }
}

void preCalcGrid(int dim, int M, int lenVec, double* vec, double* subGrid, double* preCalcGrid) {
    int i,j,k;

    for (j=0; j < dim; j++) {
        for (i=0; i < M; i++) {
            for (k=0; k < lenVec; k++) {
                preCalcGrid[i + j*M + k*dim*M] = vec[k + j*lenVec]*subGrid[i + j*M];
            }
        }
    }
}


void getSubIndex(int dim, int M, int MPow, int* subIdx) {
    int k, j, raiseIndex;

    for (j=0; j < dim; j++) {
        subIdx[j] = 0;
    }

    /* the first index always grows until M, then is reset to 0; other indices only grows if the previous one reaches M */
    for (j=1; j < MPow; j++) {
        /* increase the first one */
        raiseIndex = 0;
        if (subIdx[(j-1)*dim] < M-1) {
            subIdx[j*dim] = subIdx[(j-1)*dim]+1;
        } else {
            subIdx[j*dim] = 0;
            raiseIndex = 1;
        }
        for (k=1; k < dim; k++) {
            if (raiseIndex) {
                if (subIdx[k+(j-1)*dim] < M-1) {
                    subIdx[k+j*dim] = subIdx[k + (j-1)*dim] + 1;
                    raiseIndex = 0;
                } else {
                    subIdx[k+j*dim] = 0;
                    raiseIndex = 1;
                }
            } else {
                subIdx[k + j*dim] = subIdx[k + (j-1)*dim];
            }
        }
    }
}

void makeGridC(double *X, double *sparseGrid, unsigned char **aH, unsigned short int **YIdxSub, unsigned short int **YIdx, unsigned short int **gridToBox, unsigned short int **XToBox, unsigned short int **XToBoxOuter, int **numPointsPerBox, double **boxEvalPoints, unsigned short int **boxIDs, double *subGrid, int *subGridIdx, double *ACVH, double *bCVH, double *box, int **lenY, int **numBoxes, int dim, int lenCVH, int N, int M, int numGridPoints, int NX, double *sparseDelta) {

	/* ****************************************** */
	/* ********** VARIABLE DECLARATION ********** */
	/* ****************************************** */

	int i,j,k,l,counter,numCombs,val,tmp,assign;
	double *evalBasics = malloc((N+1)*lenCVH*dim*sizeof(double));
	double interval, tmpD;

    /* for calculation of active hyperplanes of each box --> speeds up checking if subgrid points are inside the convex hull, only have to check for active hyperplanes */
    double *evalHyper = malloc(lenCVH*sizeof(double));
    int *subIdxLocal = malloc(dim*sizeof(int));
    int *variations = malloc(dim*numGridPoints*sizeof(int));

    int *combs = calloc(pow(2,dim)*dim,sizeof(int));
    int *idxBox = calloc(pow(2,dim),sizeof(int));
    int *factor = malloc(dim*sizeof(int));

    /* variables for pre-calculating some stuff */
    double *boxTmp = calloc(dim*2,sizeof(double));
    int *gridIdx = malloc(dim*numGridPoints*sizeof(int));
    int numPointsSubIdx;

    /* variables for checking boxes and adding subgrid points */
    int maxIdx; int counterNumBoxes = 0; int pointsOutside; int outsideCVH;
    int numActiveConstraints;
    int numPointsAdded = 0; int numPointsAddedOld; int offset;
    double funcEvalLocal;

	int *YBoxMin = malloc(dim*sizeof(int));
	int *YBoxMax = malloc(dim*sizeof(int));
	double *boxMax = malloc(dim*sizeof(double));
	double *boxMaxOuter = malloc(dim*sizeof(double));
	double *boxMinOuter = malloc(dim*sizeof(double));
	double *boxMin = malloc(dim*sizeof(double));
	double epsilon = pow(10,-10);
	double yTmp[dim];

    /* create arrays that are supposed to be returned to the calling function */
	*numBoxes = malloc(sizeof(int)); **numBoxes = pow(N,dim);
	*lenY = malloc(sizeof(int)); **lenY = **numBoxes*pow(M,dim);

	*aH = calloc(**numBoxes*lenCVH,sizeof(unsigned char));
	*YIdx = malloc(**lenY*dim*sizeof(unsigned short int));
    *YIdxSub = malloc(**lenY*sizeof(unsigned short int));
    *gridToBox = malloc(**lenY*sizeof(unsigned short int));
    *XToBox = malloc(NX*sizeof(unsigned short int));
    *XToBoxOuter = malloc(NX*sizeof(unsigned short int));
	for (i=0; i < NX; i++) {
		(*XToBoxOuter)[i] = USHRT_MAX;
		(*XToBox)[i] = USHRT_MAX;
	}
	
	*boxIDs = calloc(**numBoxes,sizeof(unsigned short int));
    *numPointsPerBox = calloc(**numBoxes+1,sizeof(int));
	*boxEvalPoints = calloc(**numBoxes*3*dim,sizeof(double)); 

    /* ********************************************************************** */
    /* *********** PRECALCULATE STUFF --> SPEEDS UP MAIN FOR LOOP *********** */
    /* ********************************************************************** */

    /* create all combinations of a binary vector of length dim: 000, 001, 010, 011, ... */
    numCombs = pow(2,dim);
    for (i=1; i < numCombs; i++) {
        val = i;
        for (j=dim-1; j >= 0; j--) {
            tmp = pow(2,j);
            if (tmp <= val) {
                combs[i*dim+j] = 1;
                val = val-tmp;
            }
        }
    }

    for (j=0; j < dim; j++) {
        factor[j] = pow(N+1,j);
    }

    for (i=0; i < numCombs; i++) {
        for(j=0; j < dim; j++) {
            idxBox[i] += combs[i*dim+j]*factor[j];
        }
    }
    free(factor); free(combs);

    /* make subgrid for midpoint rule (grid inside each box) */
    for (j=0; j < dim; j++) {
        boxTmp[dim+j] = (box[j+dim] - box[j])/N;
   	}
    makeGridMidpoint(boxTmp,subGrid,M,dim);
    free(boxTmp);

    /* index for selecting the subgrid */
    numPointsSubIdx = pow(M,dim);
    getSubIndex(dim,M,numPointsSubIdx,subGridIdx);
    getSubIndex(dim,(N+1),numGridPoints,gridIdx);

    /* ************************************************************************************ */
    /* ************* MAIN LOOP: ADDS ALL POINTS LYING INSIDE THE CONVEX HULL ************** */
    /* ************************************************************************************ */

    counterNumBoxes = 0; numPointsAdded = 0;
    for (l=0; l < numGridPoints; l++) {
        /* boxes can't correspond to grid points that lie on the right boundary */
        for (i=0; i < dim; i++) {
            if (variations[i+dim*l]>maxIdx) {
                maxIdx = variations[i+dim*l];
            }
        }
        if (maxIdx==N) {
            continue;
        }

		/* reset box min and max */
		for (j=0; j < dim; j++) {
			YBoxMax[j] = 0; YBoxMin[j] = M;
		}

        numPointsAddedOld = numPointsAdded;
        /* check points in subgrid */
        for (j=0; j < numPointsSubIdx; j++) {
            outsideCVH = 0;
			for (k=0; k < dim; k++) {
				yTmp[k] = sparseGrid[l*dim + k] + subGrid[subGridIdx[k + j*dim] + k*M];
			}
			/* check each point */
            for (i=0; i < lenCVH; i++) {
                funcEvalLocal = -bCVH[i];
                for (k=0; k < dim; k++) {
                    funcEvalLocal += ACVH[i+k*lenCVH]*yTmp[k];
                }
                if (funcEvalLocal > 0) {
                    outsideCVH = 1;
                    break; /* as soon as one violating hyperplane is found, we can stop the search and continue with the next grid point */
                }
            }
            /* if inside convex hull, add point to list of points */
            if (outsideCVH==0) {
                for (k=0; k < dim; k++) {
/*                  (*Y)[k + numPointsAdded*dim] = sparseGrid[l*dim + k] + subGrid[subGridIdx[k + j*dim] + k*N]; */
                    (*YIdx)[k + numPointsAdded*dim] = subGridIdx[k + j*dim] + gridIdx[k + l*dim]*M;
					/* check for min and max */
					if (YBoxMin[k] > subGridIdx[k+j*dim]) {
						YBoxMin[k] = subGridIdx[k+j*dim];
					}
					if (YBoxMax[k] < subGridIdx[k+j*dim]) {
						YBoxMax[k] = subGridIdx[k+j*dim];
					}
                }
                /* save subGridIdx (1-D index from 1 to M^d) */
                (*YIdxSub)[numPointsAdded] = j;
                (*gridToBox)[numPointsAdded] = counterNumBoxes;
                numPointsAdded++;
		   	}
		}
		/* find box max min of outer box boundaries */
		for (j=0; j < dim; j++) {
			boxMinOuter[j] = sparseGrid[l*dim+j];
			boxMaxOuter[j] = sparseGrid[l*dim+j] + sparseDelta[j];
		}

		/* find box max and min, based on the sparse grid defining box boundaries (and not on the subgrid inside the box) 
		 * With that definition, we can use the estimates of active hyperplanes for each box also when evaluating X */
		for (j=0; j < dim; j++) {
			boxMin[j] = sparseGrid[l*dim + j]+subGrid[YBoxMin[j] + j*M]; 
			boxMax[j] = sparseGrid[l*dim + j]+subGrid[YBoxMax[j] + j*M];
		}

        /* only add the box, if at least one grid point was added */
        if (numPointsAdded > numPointsAddedOld) {
			/* iterate over all X values and check whether they belong to the current box */
			for (j=0; j < NX; j++) {
				/* if still unassigned */
				if ((*XToBoxOuter)[j] == USHRT_MAX) {
					assign = 1;
					for (k=0; k < dim; k++) {
						if (X[k*NX + j] > boxMaxOuter[k]+epsilon || X[k*NX+j] < boxMinOuter[k]-epsilon) {
							assign = 0;
							break;
						}
					}
					if (assign==1) {
						(*XToBoxOuter)[j] = counterNumBoxes;
						assign = 1;
						/* check for inner box */
						for (k=0; k < dim; k++) {
							if (X[k*NX + j] > boxMax[k]+epsilon || X[k*NX+j] < boxMin[k]-epsilon) {
								assign = 0;
								break;
							}
						}
						if (assign==1) {
							(*XToBox)[j] = counterNumBoxes;
						}
					}
				}
			}
	    	(*numPointsPerBox)[counterNumBoxes+1] = numPointsAdded;
			for (j=0; j < dim; j++) {
				/* choose the point closest to zero, as it most probably lies inside the convex hull of X */
				/* for each box three points: The point to evaluate, the delta between the two corners and the sign indicating in which corner we start */
				if (fabs(boxMin[j]) < fabs(boxMax[j])) {
					(*boxEvalPoints)[counterNumBoxes*3*dim + j] = boxMin[j];
					(*boxEvalPoints)[counterNumBoxes*3*dim + 2*dim + j] = 1;
				} else {
					(*boxEvalPoints)[counterNumBoxes*3*dim + j] = boxMax[j];
					(*boxEvalPoints)[counterNumBoxes*3*dim + 2*dim + j] = -1;
				}
			    (*boxEvalPoints)[counterNumBoxes*3*dim + dim + j] = boxMax[j]-boxMin[j];
			}
            (*boxIDs)[counterNumBoxes++] = l;
 		}
    }
    if (counterNumBoxes != **numBoxes) {
		*aH = realloc(*aH,counterNumBoxes*lenCVH*sizeof(unsigned char));
        *boxIDs = realloc(*boxIDs,counterNumBoxes*sizeof(unsigned short int));
        *numPointsPerBox = realloc(*numPointsPerBox,(counterNumBoxes+1)*sizeof(int));
		*boxEvalPoints = realloc(*boxEvalPoints,counterNumBoxes*dim*3*sizeof(double)); 
    }
    **numBoxes = counterNumBoxes;

    if (numPointsAdded != **lenY) {
        *YIdx = realloc(*YIdx,numPointsAdded*dim*sizeof(unsigned short int));
        *YIdxSub = realloc(*YIdxSub,numPointsAdded*sizeof(unsigned short int));
        *gridToBox = realloc(*gridToBox,numPointsAdded*sizeof(unsigned short int));
    }
    **lenY = numPointsAdded;

    free(idxBox);
    free(variations);
    free(gridIdx);
	free(YBoxMax); free(YBoxMin);
}
