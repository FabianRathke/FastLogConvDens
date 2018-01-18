function [aOpt bOpt logLike statistics gridParams] = lcdFast(X,gamma,optOptions)

% ********* SET SOME DEFAULT OPTIONS

if ~isfield(optOptions,'method')
	optOptions.method = @newtonBFGSLC;
end

if ~isfield(optOptions,'sampleWeights')
    optOptions.sampleWeights = ones(length(X),1);
end

if ~isfield(optOptions,'returnGrid')
	optOptions.returnGrid = 0;
end

if ~isfield(optOptions,'fast')
	optOptions.fast = 0;
end

if ~isfield(optOptions,'minHyperplanes')
	optOptions.minHyperplanes = 0;
end

if ~isfield(optOptions,'init')
	optOptions.init = '';
end

sW = optOptions.sampleWeights/sum(optOptions.sampleWeights); % normalize sampleWeights

[n dim] = size(X);

% zero mean for X
mu = mean(X);
X = X - repmat(mu,n,1);

% *************** OBTAIN GRID FOR INTEGRATION **********
%tic; [gridParams X optOptions] = obtainGrid(X,optOptions); timingGrid = toc;
[gridParams.ACVH gridParams.bCVH gridParams.cvh] = calcCvxHullHyperplanes(X);

optOptions.mu = mu;

paramsKernel = [];
if ~isfield(optOptions,'b') || ~isfield(optOptions,'a');
	% ************ FIT HYPERPLANES TO KERNEL DENSITY *********
	tic;
	% if the user specified the initialization 
	if strcmp(optOptions.init,'kernel')
		params = paramFitKernelDensity(X,sW,gridParams.cvh);
		initSelect = 'kernel';
	elseif strcmp(optOptions.init,'gamma')
		params = paramFitGammaOne(X,sW,gridParams.ACVH,gridParams.bCVH,gridParams.cvh,optOptions);
		initSelect = 'gamma';
	else
		if n < 2500
			paramsKernel = paramFitKernelDensity(X,sW,gridParams.cvh);
			params = paramFitGammaOne(X,sW,gridParams.ACVH,gridParams.bCVH,gridParams.cvh,optOptions);
			%compareInitialization;
		else
			params = paramFitGammaOne(X,sW,gridParams.ACVH,gridParams.bCVH,gridParams.cvh,optOptions);
			initSelect = 'gamma';
		end
	end
	initializeHyperplanes = toc;
else
	params = [optOptions.a(:); optOptions.b];
	initializeHyperplanes = 0;
	initSelect = 'Preset';
end

%[optParams logLike statistics] = optOptions.method(params(:),X,sW,gamma,gridParams,optOptions); statistics.timings.makeGrid = timingGrid;
optParams = bfgsFullC(X,sW,params,paramsKernel,[min(X)' max(X)'],gridParams.ACVH,gridParams.bCVH,optOptions.verbose,optOptions.intEps, optOptions.lambdaSqEps, optOptions.cutoff);

numHypers = length(optParams)/(dim+1); aOpt = optParams(1:dim*numHypers); aOpt = reshape(aOpt,[],dim); bOpt = optParams(dim*numHypers+1:end);

statistics = struct();
% project density into the valid function class and renormalize it there
[statistics aOpt bOpt logLike T yT Ad Gd] = correctIntegral(X,mu,sW,aOpt,bOpt,statistics,optOptions,gridParams.cvh);

% update convex hull parameters for true X
gridParams.bCVH = gridParams.bCVH+gridParams.ACVH*mu';
