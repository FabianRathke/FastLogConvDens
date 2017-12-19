function [params gridParams statistics] = paramFitGammaOne(X,sampleWeights,ACVH,bCVH,cvh,optOptions) 

[n dim] = size(X);

% grid params for the sparse grid used for initialization
[N M gridParams.grid gridParams.weight gridParams.gridSize] = setGridDensity([min(X)' max(X)'],dim,1,optOptions);
gridParams.N = N; gridParams.M = M;
gridParams.delta = [gridParams.grid(:,2)-gridParams.grid(:,1)];
gridParams.ACVH = ACVH; gridParams.bCVH = bCVH;
gridParams.sparseGrid = makeGridND([min(X)' max(X)'],N);
gridParams.sparseDelta = (gridParams.sparseGrid(:,end)-gridParams.sparseGrid(:,1))/gridParams.N;
[gridParams.YIdx gridParams.XToBox, gridParams.numPointsPerBox, gridParams.boxEvalPoints] = makeGrid(gridParams.sparseGrid,[min(X) max(X)],ACVH,bCVH,N,M,dim,X,gridParams.sparseDelta);

% initialize parameters randomly; for $\gamma = 1$ we are less sensitive to the initialization as we have a much more well behaved objective function
[N dim] = size(X);
n = 10*dim; lenP = n*(dim+1);
a = rand(n,dim)*0.1; b = rand(n,1);
params = [reshape(a,[],1); b];

[optParams logLike statistics] = newtonBFGSLCInit(params,X,1,gridParams,struct('verbose',0,'reduceHyperplanes',0));
aOpt = reshape(optParams(1:10*dim*dim),[],dim); bOpt = optParams(10*dim*dim+1:end);

yT = -log(sum(exp(aOpt*X' + repmat(bOpt,1,length(X)))))';

idxCVH = unique(cvh);
T = int32(convhulln([X yT; X(idxCVH,:) repmat(min(yT(idxCVH))-1,length(idxCVH),1)]));
%T = int32(convhulln([X yT; X repmat(min(yT)-1,length(yT),1)]));
T(max(T,[],2)>length(yT),:) = [];

numHypers = length(T);
[aOpt bOpt] = calcExactIntegral(X',yT,T'-1,dim,1);
aOpt = aOpt';

% restrict the initial number of hyperplanes for d=1 to 1000
if (size(X,2) == 1) && length(bOpt) > 1000
	idxSelect = randperm(length(bOpt),1000);
	aOpt = aOpt(idxSelect);
	bOpt = bOpt(idxSelect);
end

params = [aOpt bOpt];

% detect if all hyerplanes where intitialized to the same parameters; happens for small sample sizes --> very slow convergence in the full optimization
if mean(var(aOpt(randperm(length(bOpt),min(100,length(bOpt))),:))) < 10^-4 || length(params) == dim+1
	fprintf('#### Bad initialization due to small sample size, switch to kernel kensity based initialization ####\n');
	params = paramFitKernelDensity(X,optOptions.sampleWeights,gridParams,cvh);
end

