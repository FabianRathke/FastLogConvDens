function params = paramFitKernelDensity(X,sampleWeights,cvh) 
% fits a piecewise linear polynomial to a kernel density estimate of X at positions Y (adapted from Cule et al.)
%
% [params] = paramsFitKernelDensity(X,Y,grid,numHypers,sampleWeights)
timeKernel = tic;
[n dim] = size(X);
yT = log(kernelDens(X,sampleWeights));

infVals = ~isinf(yT);
%T = int32(convhulln([X(infVals,:) yT(infVals); X(infVals,:) repmat(min(yT(infVals))-1,sum(infVals),1)]));
idxCVH = setdiff(unique(cvh),find(~infVals));
T = int32(convhulln([X(infVals,:) yT(infVals); X(idxCVH,:) repmat(min(yT(idxCVH))-1,length(idxCVH),1)]));
T(max(T,[],2)>sum(infVals),:) = [];

numHypers = length(T);
[aOpt bOpt] = calcExactIntegral(X(infVals,:)',yT(infVals),T'-1,dim,1,10^-2); 
params = [aOpt' bOpt];
toc(timeKernel);
