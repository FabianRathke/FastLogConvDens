function params = paramFitKernelDensity(X,sampleWeights,cvh,verbose) 
% fits a piecewise linear polynomial to a kernel density estimate of X at positions Y (adapted from Cule et al.)
%
% [params] = paramsFitKernelDensity(X,Y,grid,numHypers,sampleWeights)
%timeKernel = tic;
[n dim] = size(X);
h = std(X)*n^(-1/(dim+4));
yT = log(kernelDensC(X',sampleWeights,h));

finiteVals = ~isinf(yT);
idxCVH = setdiff(unique(cvh),find(~finiteVals));
if verbose > 1
	fprintf('Calculate convex hull\n');
end
T = int32(convhulln([X(finiteVals,:) yT(finiteVals); X(idxCVH,:) repmat(min(yT(idxCVH))-1,length(idxCVH),1)]));
T(max(T,[],2)>sum(finiteVals),:) = [];

if verbose > 1
	fprintf('Normalize integral for kernel density estimate\n');
end
numHypers = length(T);
[aOpt bOpt] = calcExactIntegral(X(finiteVals,:)',yT(finiteVals),T'-1,dim,1,10^-2); 
params = [aOpt' bOpt];
if verbose > 1
	fprintf('Finished normalizing intial kernel density\n');
end
%toc(timeKernel);
