#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	float *s_k = (float *) mxGetData(prhs[0]);
   	float *y_k = (float *) mxGetData(prhs[1]);
	float *sy = (float *) mxGetData(prhs[2]);
    float *syInv = (float *) mxGetData(prhs[3]);
    float step = (float)  mxGetScalar(prhs[4]);
    float *grad = (float *) mxGetData(prhs[5]);
    float *gradOld = (float *) mxGetData(prhs[6]);
   	float *newtonStep = (float *) mxGetData(prhs[7]);
	int numIter = (int) mxGetScalar(prhs[8]); // iteration number of the newtonBFGS scheme --> helps to determine the number of stored vectors in s_k
    int activeCol = (int) mxGetScalar(prhs[9]); // activeCol should be C-indexing
 
	int nH = mxGetNumberOfElements(prhs[5]); /* number of hyperplanes*(dim+1) */
	int m = mxGetN(prhs[0]);

    const int dimsA[] = {nH};
	plhs[0] = mxCreateNumericArray(1,dimsA,mxSINGLE_CLASS,mxREAL);
	float* newtonStepNew = (float *) mxGetData(plhs[0]);

	float normTmp, dotProd, dotProd2;
	float s_k_tmp, y_k_tmp, t, H0;
	float* gammaBFGS = (float *) malloc(nH*sizeof(float));
	float* q = (float *) malloc(nH*sizeof(float));
	float alphaBFGS[numIter];
	float betaBFGS,tmp;
	int iterVec[numIter];
	int activeIdx = activeCol*nH; 	
	int j,l,curCol;

	normTmp = dotProd = t = 0;
	
	#pragma omp parallel for reduction(+:normTmp,dotProd,t) private(s_k_tmp)
	for (j=0; j < nH; j++) {
		gammaBFGS[j] = grad[j] - gradOld[j];
		s_k_tmp = step*newtonStep[j];
		dotProd += gammaBFGS[j]*s_k_tmp;
		normTmp += s_k_tmp*s_k_tmp;
		t += gradOld[j]*gradOld[j];
		s_k[activeIdx + j] = s_k_tmp;
	}

	t = sqrtf(t); // finish calculation of norm
	if (-dotProd/normTmp > 0) {
		t += -dotProd/normTmp;
	}
//	printf("t: %.7f, gammaBFGS'*s_k: %.4e, s_k'*s_k: %.4e\n",t,dotProd,normTmp);

	dotProd = dotProd2 = 0;
	#pragma omp parallel for reduction(+:dotProd,dotProd2) private(y_k_tmp)
	for (j=0; j < nH; j++) {
		y_k_tmp = gammaBFGS[j] + t*s_k[activeIdx+j];
		y_k[activeIdx + j] = y_k_tmp;
		dotProd += y_k_tmp*s_k[activeIdx+j];
		dotProd2 += y_k_tmp*y_k_tmp;
	}
	sy[activeCol] = dotProd; 
	syInv[activeCol] = 1/sy[activeCol];

    H0 = sy[activeCol]/dotProd2;
//	printf("H0: %.4f, sy: %.4e, syInv: %.4e, y_k'*y_k: %.3e\n",H0,sy[activeCol],syInv[activeCol],dotProd2);

//	printf("y_k: %.4e, s_k: %.4e\n",y_k[activeIdx],s_k[activeIdx]);
	for (j=0; j < numIter; j++) {
		iterVec[j] = activeCol-j;
		if (iterVec[j] < 0) {
			iterVec[j] = m + iterVec[j];
		}
	}
	for (j=0; j < numIter; j++) {
//		printf("Iter %d = %d\n",j,iterVec[j]);
	}
//	printf("syInv: %.4f\n",syInv[0]);
	// q = grad
	memcpy(q,grad,nH*sizeof(float));
    // first for-loop
	for (l=0; l < numIter; l++) {
		curCol = iterVec[l];
		tmp = 0;
		#pragma omp parallel for reduction(+:tmp)
		for (j=0; j < nH; j++) {
			tmp += s_k[curCol*nH + j]*q[j];
		}
		alphaBFGS[curCol] = tmp*syInv[curCol];
//	   	printf("alphaBFGS: %.4f\n",alphaBFGS[curCol]);
		#pragma omp parallel for
		for (j=0; j < nH; j++) {
			q[j] -= alphaBFGS[curCol]*y_k[curCol*nH+j];
		}
	}


	for (j=0; j < nH; j++) {
		q[j] = H0*q[j]; // is "r" in the matlab code and the book
	}
	for (j = 0; j < 10; j++) {
//		printf("q[%d]: %.4e\n",j,q[j]);
	}
	// second for-loop
	for (l=0; l < numIter; l++) {
		curCol = iterVec[numIter-1-l];
		betaBFGS = 0;
		#pragma omp parallel for reduction(+:betaBFGS)
		for (j=0; j < nH; j++) {
			betaBFGS += y_k[curCol*nH+j]*q[j];
		}
		betaBFGS *= syInv[curCol];
//		printf("betaBFGS: %.4f\n",betaBFGS);
		tmp = (alphaBFGS[curCol]-betaBFGS);
		#pragma omp parallel for
		for (j=0; j < nH; j++) {
			q[j] += s_k[curCol*nH + j]*tmp;
		}
	}
	for (j = 0; j < 10; j++) {
		//printf("q[%d]: %.4e\n",j,q[j]);
	}
	

	for (j=0; j < nH; j++) {
		newtonStepNew[j] = -q[j];
	}
	free(gammaBFGS); free(q);
}
