void calcGradFloatC(float* grad, float* influence, float* TermA, float* TermB, float* X, float* XW, float* grid, unsigned short int* YIdx, float* a, float* b, float gamma, float weight, float* delta, int N, int M, int dim, int nH, int n);
void calcGradFullAVXC(float* grad, float* influence, float* TermA, float* TermB, float* X, float* XW, float* grid, unsigned short int* YIdx, float* a, float* b, float gamma, float weight, float* delta, int N, int M, int dim, int nH);
void calcGradC(double* gradA, double* gradB, double* influence, double* TermA, double* TermB, double* X, double* XW, double* grid, unsigned short int* YIdx, int *numPointsPerBox, double* boxEvalPoints, unsigned short int *XToBox, int numBoxes, double* a, double* b, double gamma, double weight, double* delta, int N, int M, int dim, int nH, int MBox, double* evalFunc);
void calcGradMaxC(double* gradA, double* gradB, double* influence, double* TermA, double* TermB, double* X, double* XW, double* grid, unsigned short int* YIdx, int *numPointsPerBox, double* boxEvalPoints, unsigned short int *XToBox, int numBoxes, double* a, double* b, double weight, double* delta, int N, int M, int dim, int nH, int n, int MBox, double* evalFunc);
void calcGradFastC(int* numEntries, int* elementList, int* maxElement, int* idxEntries, double* grad, double* influence, double* TermA, double* TermB, double* X, double* XW, double* grid, unsigned short int* YIdx, double* a, double* b, double gamma, double weight, double* delta, int N, int M, int dim, int nH);
void calcGradFastMaxC(int* numEntries, int* elementList, int* maxElement, int* idxEntries, double* grad, double* influence, double* TermA, double* TermB, double* X, double* XW, double* grid, unsigned short int* YIdx, double* a, double* b, double weight, double* delta, int N, int M, int dim, int nH, int n, double* evalGrid);
void preCondGradC(int** elementList, int** elementListSize, int* numEntries, int* maxElement, int* idxEntries, double* X, double* grid, unsigned short int* YIdx, int *numPointsPerBox, double* boxEvalPoints, int numBoxes, double* a, double* b, double gamma, double weight, double* delta, int N, int M, int dim, int nH, int n, int MBox);
void getSubIndex(int dim, int M, int MPow, int* subIdx);
void preCalcGrid(int dim, int M, int lenVec, double* vec, double* subGrid, double* preCalcGrid);
double calcExactIntegralC(double* X, double* y, int* T, int lenT, int lenY, int dim, double* integral, double targetIntegral, double intEps, double* changeB, double* Ad, double* Gd);
void recalcParamsC(double* X, double* y, int* T, int lenT, int dim, double* aOptNew, double* bOptNew);
void evalObjectiveC(double* X, double* XW, double* grid, unsigned short int* YIdx, int *numPointsPerBox, double* boxEvalPoints, int MBox, int numBoxes, double* a, double* b, double gamma, double weight, double* delta, int N, int M, int dim, int nH, int n, double* evalFunc);
void samplesLCDC(double* X, int n, int* T, double* yT, int * samplesSimplex, double* samples, double* samplesEval, int N, int dim, int lenT);

