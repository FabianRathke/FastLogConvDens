function [aOpt bOpt logLike gridParams] = lcdFast(X,gamma,optOptions)

% ********* SET SOME DEFAULT OPTIONS

if ~isfield(optOptions,'sampleWeights')
    optOptions.sampleWeights = ones(length(X),1);
end

if ~isfield(optOptions,'init')
	optOptions.init = '';
end

sW = optOptions.sampleWeights/sum(optOptions.sampleWeights); % normalize sampleWeights

[n dim] = size(X);

% zero mean for X
mu = mean(X);
X = X - repmat(mu,n,1);

% the faces of the convex hull of X
[gridParams.ACVH gridParams.bCVH gridParams.cvh] = calcCvxHullHyperplanes(X);

paramsKernel = [];
if ~isfield(optOptions,'b') || ~isfield(optOptions,'a');
	% ************ FIT HYPERPLANES TO KERNEL DENSITY *********
	% if the user specified the initialization 
	if strcmp(optOptions.init,'kernel')
		params = paramFitKernelDensity(X,sW,gridParams.cvh);
		initSelect = 'kernel';
	elseif strcmp(optOptions.init,'gamma')
		params = paramFitGammaOne(X,sW,gridParams.ACVH,gridParams.bCVH,gridParams.cvh,optOptions);
	else
		if n < 2500
			paramsKernel = paramFitKernelDensity(X,sW,gridParams.cvh);
			params = paramFitGammaOne(X,sW,gridParams.ACVH,gridParams.bCVH,gridParams.cvh,optOptions);
		else
			params = paramFitGammaOne(X,sW,gridParams.ACVH,gridParams.bCVH,gridParams.cvh,optOptions);
		end
	end
else
	params = [optOptions.a(:); optOptions.b];
end

optParams = bfgsFullC(X,sW,params,paramsKernel,[min(X)' max(X)'],gridParams.ACVH,gridParams.bCVH,optOptions.verbose,optOptions.intEps, optOptions.lambdaSqEps, optOptions.cutoff);
numHypers = length(optParams)/(dim+1); aOpt = optParams(1:dim*numHypers); aOpt = reshape(aOpt,[],dim); bOpt = optParams(dim*numHypers+1:end);

% project density into the valid function class and renormalize it there
[aOpt bOpt T yT Ad Gd] = correctIntegral(X,mu,aOpt,bOpt,optOptions,gridParams.cvh);
logLike = yT.*sW'*length(sW);

% update convex hull parameters for true X
gridParams.bCVH = gridParams.bCVH+gridParams.ACVH*mu';
