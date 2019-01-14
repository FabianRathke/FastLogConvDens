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
	minGridSize = [1000, 5000, 25000, 40000, 60000, 80000, 100000, 120000, 140000];
	optOptions.minGridSize = minGridSize(dim);
	optOptions.minGridSizeInit = round(optOptions.minGridSize/10);
end

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
		params = paramFitGammaOne(X,sW,gridParams.ACVH,gridParams.bCVH,gridParams.cvh,optOptions.gammaInit,ratio, optOptions.minGridSizeInit);
	else
		if n < 2500
			params = paramFitGammaOne(X,sW,gridParams.ACVH,gridParams.bCVH,gridParams.cvh,optOptions.gammaInit,ratio, optOptions.minGridSizeInit);
			paramsKernel = paramFitKernelDensity(X,sW,gridParams.cvh);
		else
			params = paramFitGammaOne(X,sW,gridParams.ACVH,gridParams.bCVH,gridParams.cvh,optOptions.gammaInit,ratio, optOptions.minGridSizeInit);
		end
	end
else
	params = [optOptions.a(:); optOptions.b];
end

% minimal grid sizes

[optParams gridParams.YIdx, gridParams.grid, gridParams.weight] = bfgsFullC(X,sW,params,paramsKernel,[min(X)' max(X)'],gridParams.ACVH,gridParams.bCVH,optOptions.verbose,optOptions.intEps, optOptions.lambdaSqEps, optOptions.cutoff, gamma, ratio, optOptions.minGridSize);
%[optParams] = bfgsFullC(X,sW,params,paramsKernel,[min(X)' max(X)'],gridParams.ACVH,gridParams.bCVH,optOptions.verbose,optOptions.intEps, optOptions.lambdaSqEps, optOptions.cutoff);
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
