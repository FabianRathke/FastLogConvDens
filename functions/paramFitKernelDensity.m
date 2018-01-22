function params = paramFitKernelDensity(X,sampleWeights,cvh) 
% fits a piecewise linear polynomial to a kernel density estimate of X at positions Y (adapted from Cule et al.)
%
% [params] = paramsFitKernelDensity(X,Y,grid,numHypers,sampleWeights)
%timeKernel = tic;
[n dim] = size(X);
h = std(X)*n^(-1/(dim+4));
yT = log(kernelDensC(X',sampleWeights,h));

finiteVals = ~isinf(yT);
idxCVH = setdiff(unique(cvh),find(~finiteVals));
T = int32(convhulln([X(finiteVals,:) yT(finiteVals); X(idxCVH,:) repmat(min(yT(idxCVH))-1,length(idxCVH),1)]));
T(max(T,[],2)>sum(finiteVals),:) = [];

numHypers = length(T);
[aOpt bOpt] = calcExactIntegral(X(finiteVals,:)',yT(finiteVals),T'-1,dim,1,10^-2); 
params = [aOpt' bOpt];
%toc(timeKernel);
