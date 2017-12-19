function [aOpt bOpt logLike statistics gridParams seed] = lcdFast(X,gamma,optOptions)

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

optOptions.sampleWeights = optOptions.sampleWeights/sum(optOptions.sampleWeights); % normalize sampleWeights

[n dim] = size(X);

% zero mean for X
mu = mean(X);
X = X - repmat(mu,n,1);

% *************** OBTAIN GRID FOR INTEGRATION **********
tic; [gridParams X optOptions] = obtainGrid(X,optOptions); timingGrid = toc;
optOptions.mu = mu;

if ~isfield(optOptions,'b')
	% ************ FIT HYPERPLANES TO KERNEL DENSITY *********
	tic;
	% for small sample sizes the kernel initialization could perform better
	if strcmp(optOptions.init,'kernel')
		params = paramFitKernelDensity(X,optOptions.sampleWeights,gridParams,gridParams.cvh);
		initSelect = 'kernel';
	elseif strcmp(optOptions.init,'gamma')
		[params gridParamsInit statisticsInit] = paramFitGammaOne(X,optOptions.sampleWeights,gridParams.ACVH,gridParams.bCVH,gridParams.cvh,optOptions);
		initSelect = 'gamma';
	else
		if n < 2500
			paramsKernel = paramFitKernelDensity(X,optOptions.sampleWeights,gridParams,gridParams.cvh);
			[params gridParamsInit statisticsInit] = paramFitGammaOne(X,optOptions.sampleWeights,gridParams.ACVH,gridParams.bCVH,gridParams.cvh,optOptions);
			compareInitialization;
		else
			[params gridParamsInit statisticsInit] = paramFitGammaOne(X,optOptions.sampleWeights,gridParams.ACVH,gridParams.bCVH,gridParams.cvh,optOptions);
			initSelect = 'gamma';
		end
	end
	initializeHyperplanes = toc;
else
	params = [optOptions.a(:); optOptions.b];
	initializeHyperplanes = 0;
	statisticsInit.timings = struct();
	initSelect = 'Preset';
end

if optOptions.verbose > 0
	if strcmp(initSelect,'gamma')
		fprintf('Initialized using a smooth log-concave density with gamma = 1\n');
	elseif strcmp(initSelect,'kernel')
		fprintf('Initialized using kernel density\n');
	else
		fprintf('Initialized using predefined hyperplanes\n');
	end
end


if optOptions.verbose > 1
	fprintf('******* Run optimization on %d grid points and %d hyperplanes ***********\n',length(gridParams.YIdx),length(params(:))/(dim+1));
end

[optParams logLike statistics] = optOptions.method(params(:),X,gamma,gridParams,optOptions); statistics.timings.makeGrid = timingGrid;
numHypers = length(optParams)/(dim+1); aOpt = optParams(1:dim*numHypers); aOpt = reshape(aOpt,[],dim); bOpt = optParams(dim*numHypers+1:end);

% project density into the valid function class and renormalize it there
[statistics aOpt bOpt logLike T yT Ad Gd] = correctIntegral(X,mu,optOptions.sampleWeights,aOpt,bOpt,statistics,optOptions,gridParams.cvh);
gridParams.T = T; gridParams.yT = yT; gridParams.Ad = Ad; gridParams.Gd = Gd;

statistics.timings.initializeHyperplanes = initializeHyperplanes;
if strcmp(initSelect,'gamma')
	statistics.timings.init = statisticsInit.timings;
end
statistics.numGridPoints = length(gridParams.YIdx);
statistics.initSelect = initSelect;

% update convex hull parameters for true X
X = X + repmat(mu,n,1);
%[gridParams.ACVH gridParams.bCVH cvh] = calcCvxHullHyperplanes(X,gridParams.cvh);
gridParams.bCVH = gridParams.bCVH+gridParams.ACVH*mu';

% shift grid to accompany for mean shift
gridParams.grid = gridParams.grid+repmat(mu',1,size(gridParams.grid,2));
gridParams.sparseGrid = gridParams.sparseGrid + repmat(mu',1,size(gridParams.sparseGrid,2));
gridParams.subGrid = gridParams.subGrid + repmat(mu,size(gridParams.subGrid,1),1);
gridParams.boxEvalPoints(:,1:3:end) = gridParams.boxEvalPoints(:,1:3:end)+repmat(mu',1,length(gridParams.boxIDs));
gridParams.X = X;
