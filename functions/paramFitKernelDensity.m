function params = paramFitKernelDensity(X,sampleWeights,cvh) 
% fits a piecewise linear polynomial to a kernel density estimate of X at positions Y (adapted from Cule et al.)
%
% [params] = paramsFitKernelDensity(X,Y,grid,numHypers,sampleWeights)
%timeKernel = tic;
[n dim] = size(X);
h = std(X)*n^(-1/(dim+4));
yT = log(kernelDensC(X',sampleWeights,h));

infVals = ~isinf(yT);
idxCVH = setdiff(unique(cvh),find(~infVals));
T = int32(convhulln([X(infVals,:) yT(infVals); X(idxCVH,:) repmat(min(yT(idxCVH))-1,length(idxCVH),1)]));
T(max(T,[],2)>sum(infVals),:) = [];

numHypers = length(T);
[aOpt bOpt] = calcExactIntegral(X(infVals,:)',yT(infVals),T'-1,dim,1,10^-2); 
params = [aOpt' bOpt];
%toc(timeKernel);
