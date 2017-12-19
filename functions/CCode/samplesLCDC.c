#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <float.h>
#include <time.h>

extern void srand48(long int seedval);
extern double drand48(void);

void samplesLCDC(double* X, int n, int* T, double* yT, int * samplesSimplex, double* samples, double* samplesEval, int N, int dim, int lenT)
{
	int j;
	#pragma omp parallel num_threads(NUMCORES)
	{
		double *w = malloc((dim+1)*sizeof(double));
		double fw, fwMax, y;
		int d,e,samSim;
		double num, sumW,sampleVal;
		long int timeVal;
		timeVal = (long int) time(NULL);
		srand48((long) (omp_get_thread_num()+1)*timeVal);
		#pragma omp for
		for (j = 0; j < n; j++) {
			samSim = samplesSimplex[j];
			while (samples[j*dim]==0) {
				sumW = 0; 
				/* draw exponential random numbers with rate 1 */
				for (d = 0; d < dim+1; d++) {
					num = drand48();
					w[d] = -log(1-num);
					sumW += w[d];
				}
				fw = 0; fwMax = -DBL_MAX;
				for (d=0; d < dim+1; d++) {
					w[d] = w[d]/sumW;
					y = yT[T[samSim + lenT*d]];
					fw += y*w[d];
					if (y > fwMax) {
						fwMax = y;
					}
				}
				fw = exp(fw);
				fwMax = exp(fwMax);
				if (drand48() < fw/fwMax) {
					for (d=0; d < dim; d++) {
						sampleVal = 0;
						for (e=0; e < dim+1; e++) {
							sampleVal += w[e]*X[T[samSim + lenT*e] + N*d];
						}
						samples[j*dim + d] = sampleVal;
					}
					samplesEval[j] = fw;
				}
			}
		}
	}
}
