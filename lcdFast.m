function [aOpt bOpt logLike gridParams statistics] = lcdFast(X,gamma,optOptions)

[n dim] = size(X);

% ********* SET SOME DEFAULT OPTIONS

if ~isfield(optOptions,'sampleWeights')
    optOptions.sampleWeights = ones(length(X),1);
end

if ~isfield(optOptions,'init')
	optOptions.init = '';
end

if ~isfield(optOptions, 'gammaInit')
	optOptions.gammaInit = 1;
end

if ~isfield(optOptions, 'minGridSize')
	minGridSize = [1000, 7500, 15000, 30000, 40000, 40000, 50000, 60000, 70000];
	optOptions.minGridSize = minGridSize(dim);
end
optOptions.minGridSizeInit = round(optOptions.minGridSize/3);
sW = optOptions.sampleWeights/sum(optOptions.sampleWeights); % normalize sampleWeights

% zero mean for X
mu = mean(X);
X = X - repmat(mu,n,1);

% the faces of the convex hull of X
[gridParams.ACVH gridParams.bCVH gridParams.cvh, gridParams.V] = calcCvxHullHyperplanes(X);
ratio = gridParams.V/prod(max(X)-min(X));

paramsKernel = [];
if ~isfield(optOptions,'b') || ~isfield(optOptions,'a');
	% ************ FIT HYPERPLANES TO KERNEL DENSITY *********
	% if the user specified the initialization 
	if strcmp(optOptions.init,'kernel')
		params = paramFitKernelDensity(X,sW,gridParams.cvh);
		initSelect = 'kernel';
	elseif strcmp(optOptions.init,'gamma')
		params = paramFitGammaOne(X,sW,gridParams.ACVH,gridParams.bCVH,gridParams.cvh,optOptions.gammaInit,ratio, optOptions.minGridSizeInit,optOptions.verbose);
	else
		if n < 2500
			params = paramFitGammaOne(X,sW,gridParams.ACVH,gridParams.bCVH,gridParams.cvh,optOptions.gammaInit,ratio, optOptions.minGridSizeInit,optOptions.verbose);
			paramsKernel = paramFitKernelDensity(X,sW,gridParams.cvh,optOptions.verbose);
		else
			params = paramFitGammaOne(X,sW,gridParams.ACVH,gridParams.bCVH,gridParams.cvh,optOptions.gammaInit,ratio, optOptions.minGridSizeInit,optOptions.verbose);
		end
	end
else
	params = [optOptions.a(:); optOptions.b];
end

[optParams gridParams.YIdx, gridParams.grid, gridParams.weight] = bfgsFullC(X,sW,params,paramsKernel,[min(X)' max(X)'],gridParams.ACVH,gridParams.bCVH,optOptions.verbose,optOptions.intEps, optOptions.lambdaSqEps, optOptions.cutoff, gamma, ratio, optOptions.minGridSize);
numHypers = length(optParams)/(dim+1); aOpt = optParams(1:dim*numHypers); aOpt = reshape(aOpt,[],dim); bOpt = optParams(dim*numHypers+1:end);
statistics.numHypers = numHypers;

gridParams.aSparse = aOpt; gridParams.bSparse = bOpt;
% project density into the valid function class and renormalize it there
[aOpt bOpt T yT Ad Gd] = correctIntegral(X,mu,aOpt,bOpt,gridParams.cvh);
logLike = yT.*sW'*length(sW);

% update convex hull parameters for true X
gridParams.bCVH = gridParams.bCVH+gridParams.ACVH*mu';
% update grid
gridParams.grid = gridParams.grid + repmat(mu, size(gridParams.grid,1), 1);
