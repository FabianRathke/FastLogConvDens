#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mat.h>
#include <sys/time.h>
#include <omp.h>
#include <float.h>
#include <math.h>
#include <immintrin.h>
#include "avx_mathfun.h"

#define ALIGN 32

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void inline evalHyperplane(float* aGamma, float* bGamma, int* numElements, int* idxElements, float* ftInner, float* XAligned, int dim, int nH, int N, __m256* sum_ft, __m256* ftMax) {
	int idxA, i, k;
	__m256 ft, cmp, a, x, val1;
	__m256 delta = _mm256_set1_ps(-25);

	*numElements = 0;
	*ftMax = _mm256_set1_ps(-FLT_MAX);
	for (i=0; i < nH; i++) {
		idxA = i*dim; 
		ft = _mm256_set1_ps(*(bGamma + i));
		// ftTmp = bGamma[i] + aGamma[idxA]*X[idxB];
		for (k=0; k < dim; k++) {
			x = _mm256_load_ps(XAligned + N*k); // pull values from X for one dimension and 8 values
			a = _mm256_set1_ps(*(aGamma + idxA + k)); // fill a with the same scalar
#ifdef __AVX2__
            ft = _mm256_fmadd_ps(x,a,ft); // combined multiply+add
#else
           	ft = _mm256_add_ps(ft,_mm256_mul_ps(x,a));
#endif
		}

		// ftTmp > ftInnerMax - 25
		val1 = _mm256_add_ps(*ftMax,delta);
		cmp =_mm256_cmp_ps(ft,val1,_CMP_GT_OQ);

		// check if any value surpasses the maximum
		if (_mm256_movemask_ps(cmp) > 0) {
			// update max vals
			*ftMax = _mm256_max_ps(ft,*ftMax);
			// save ft values for later use
			_mm256_store_ps(ftInner + 8*(*numElements), ft);
			idxElements[(*numElements)++] = i;
		}
	}
	// set all values to zero
	*sum_ft = _mm256_setzero_ps();
	// calculate exp(ft) and sum_ft
	for (i=0; i < *numElements; i++) {
		ft = _mm256_load_ps(ftInner + 8*i);
		ft = exp256_ps(_mm256_sub_ps(ft,*ftMax)); // ftInner[i] = exp(ftInner[i]-ftInnerMax);
		_mm256_store_ps(ftInner + 8*i,ft);
		*sum_ft = _mm256_add_ps(*sum_ft,ft); // sum_ft += ftInner[i];
	}
}


void inline calcGradient(int numElements, int* idxElements, float* ftInner, float* XAligned, float* grad_ft_private, int dim, int nH, int N, __m256 sum_ft_inv) {
	float *t; 
	__m256 val1,ft,x;
	int i,k,idxSave;
	for (i=0; i < numElements; i++) {
		idxSave = idxElements[i];
		ft = _mm256_load_ps(ftInner + 8*i);
		ft = _mm256_mul_ps(ft,sum_ft_inv);
		for (k=0; k < dim; k++) {
			x = _mm256_load_ps(XAligned + N*k);
			val1 = _mm256_mul_ps(ft,x);
			t = (float*) &val1;
			grad_ft_private[idxSave + (k*nH)] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
		}
		t = (float*) &ft;
		grad_ft_private[idxSave + (dim*nH)] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
	}
}

void inline calcInfluence(int numElements, int* idxElements, float* ftInner, float* influencePrivate, __m256 sum_ft_inv2) {
	float* t;
	__m256 ft;
	int i, idxSave;
	for (i=0; i < numElements; i++) {
		idxSave = idxElements[i];
		ft = _mm256_load_ps(ftInner + 8*i);
		ft = _mm256_mul_ps(ft,sum_ft_inv2); //st[i]*sum_st_inv2
		t = (float*) &ft;
		influencePrivate[idxSave] += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
	}
}


void calcGradFullAVXC(float* gradA, float* gradB, double* influence, float* TermA, float* TermB, float* X, float* XW, float* grid, unsigned short int* YIdx, float* a, float* b, float gamma, float weight, float* delta, int N, int M, int dim, int nH)
{
	/* create aligned memory */
/*	float *aGamma = memalign(ALIGN,dim*nH*sizeof(float));
	float *bGamma = memalign(ALIGN, nH*sizeof(float));
	float *XAligned = memalign(ALIGN, N*dim*sizeof(float));
	float *XWAligned = memalign(ALIGN, N*sizeof(float));*/
	float *aGamma, *bGamma, *XAligned, *XWAligned;
	int errorVal = 0;
	errorVal += posix_memalign((void **) &aGamma, ALIGN, dim*nH*sizeof(float));
	errorVal += posix_memalign((void **) &bGamma, ALIGN, nH*sizeof(float));
	errorVal += posix_memalign((void **) &XAligned, ALIGN, N*dim*sizeof(float));
	errorVal += posix_memalign((void **) &XWAligned, ALIGN, N*sizeof(float));
	if (errorVal != 0) { printf("Error when using posix_memalign\n"); }

	memcpy(XAligned,X,N*dim*sizeof(float));
	memcpy(XWAligned,XW,N*sizeof(float));
    int i,j,k;
    float factor = 1/gamma, TermALocal, TermBLocal;
	int countInner;

    /* initialize some variables */
    for (i=0; i < nH; i++) {
        for (k=0; k < dim; k++) {
            aGamma[i*dim + k] = gamma*a[i*dim + k];
        }
        bGamma[i] = gamma*b[i];
    }
   	TermALocal = 0; 
	countInner = 0;
	
	// sets number of threads to the number of available threads (a less agressive option would be the number of cores: omp_get_max_threads())
	//omp_set_num_threads(omp_get_num_procs());

    #pragma omp parallel
    {   
        /* const int nthreads = omp_get_num_threads();
         * printf("Number of threads: %d\n",nthreads); */
        float *ftInner = memalign(ALIGN,8*nH*sizeof(float));
        float *grad_ft_private = calloc(nH*(dim+1),sizeof(float));
     	int *idxElements = malloc(nH*sizeof(int));
        int numElements, idxSave;
		float *t;
		__m256 ftMax,sum_ft,xw,factor_,val1,val2,sum_ft_inv,ones;
		factor_ = _mm256_set1_ps(factor);
		ones = _mm256_set1_ps(1);
        /* calculate gradient for samples X */
        #pragma omp for schedule(dynamic) private(i) reduction(+:TermALocal)
        for (j=0; j < N-(N%8); j+=8) {
			evalHyperplane(aGamma, bGamma, &numElements, idxElements, ftInner, XAligned + j, dim, nH, N, &sum_ft, &ftMax);

			xw = _mm256_load_ps(XWAligned + j);
			val1 = _mm256_add_ps(ftMax,log256_ps(sum_ft)); //(ftInnerMax + log(sum_ft))
			val2 = _mm256_mul_ps(_mm256_mul_ps(xw,val1),factor_); //*XW[j]*factor
			t = (float*) &val2;
			TermALocal += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];

            sum_ft_inv = _mm256_mul_ps(_mm256_div_ps(ones,sum_ft),xw); // sum_ft_inv = 1/sum_ft*XW[j]

			calcGradient(numElements, idxElements, ftInner, XAligned + j, grad_ft_private, dim, nH, N, sum_ft_inv);
        }
        #pragma omp critical
        {
            for (i=0; i < nH*(dim+1); i++) {
                gradA[i] += grad_ft_private[i];
            }
        }
        free(ftInner); free(grad_ft_private); free(idxElements);
    }

	TermBLocal = 0;
    #pragma omp parallel 
    {   
        float *ftInner = memalign(ALIGN,8*nH*sizeof(float));
        float *grad_ft_private = calloc(nH*(dim+1),sizeof(float));
     	int *idxElements = malloc(nH*sizeof(int));
        int numElements, idxSave;
		float *t;
		float *Ytmp = memalign(ALIGN,8*dim*sizeof(float));
		float *influencePrivate = calloc(nH,sizeof(float));
		__m256 ftMax,sum_ft,xw,factor_,factorNeg_,val1,val2,sum_ft_inv,sum_ft_inv2,ones,stInnerCorrection;
		factor_ = _mm256_set1_ps(factor);
		factorNeg_ = _mm256_set1_ps(-factor);
		ones = _mm256_set1_ps(1);
        // calculate gradient for grid points
        #pragma omp for schedule(dynamic) private(i,k) reduction(+:TermBLocal)
        for (j=0; j < M-(M%8); j+=8) { 
			for (k=0; k < dim; k++) {
				for (i=0; i < 8; i++) {
    	            Ytmp[i + k*8] = grid[k]+delta[k]*YIdx[k + (j+i)*dim];
        	    }
			}
			evalHyperplane(aGamma, bGamma, &numElements, idxElements, ftInner, Ytmp, dim, nH, 8, &sum_ft, &ftMax);
			stInnerCorrection = exp256_ps(_mm256_mul_ps(ftMax,factorNeg_)); // stInnerCorrection = exp(stInnerMax*-factor);
			val1 = _mm256_mul_ps(exp256_ps(_mm256_mul_ps(log256_ps(sum_ft),factorNeg_)),stInnerCorrection);  // tmpVal = pow(sum_st,-factor)*stInnerCorrection;

			t = (float *) &val1;
            TermBLocal += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7]; // evalGrid[j] = tmpVal;
            sum_ft_inv2 = _mm256_div_ps(ones,sum_ft); // sum_st_inv = 1/sum_st;
            sum_ft_inv = _mm256_mul_ps(val1,sum_ft_inv2); // sum_st_inv2 = tmpVal*sum_st_inv;

			calcGradient(numElements, idxElements, ftInner, Ytmp, grad_ft_private, dim, nH, 8, sum_ft_inv);
			calcInfluence(numElements, idxElements, ftInner, influencePrivate, sum_ft_inv2); 
        }
        #pragma omp critical
        {   
            for (i=0; i < nH; i++) {
                influence[i] += (double) influencePrivate[i];
            }
            for (i=0; i < nH*(dim+1); i++) {
                gradB[i] -= grad_ft_private[i]*weight;
            }
        }
        free(ftInner); free(grad_ft_private); free(idxElements); free(influencePrivate); free(Ytmp);
    }
	TermBLocal *= weight;

	*TermA += TermALocal;
	*TermB += TermBLocal;

	free(bGamma); free(aGamma); free(XAligned); free(XWAligned);
}
