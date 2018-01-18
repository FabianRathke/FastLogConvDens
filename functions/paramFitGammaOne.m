function params = paramFitGammaOne(X,sampleWeights,ACVH,bCVH,cvh,optOptions) 

[n dim] = size(X);
timeInit = tic; 
m = 10*dim;

% initialize parameters randomly; for $\gamma = 1$ we are less sensitive to the initialization as we have a much more well behaved objective function

%% grid params for the sparse grid used for initialization
%[N M gridParams.grid gridParams.weight gridParams.gridSize] = setGridDensity([min(X)' max(X)'],dim,1,optOptions);
%gridParams.N = N; gridParams.M = M;
%gridParams.delta = [gridParams.grid(:,2)-gridParams.grid(:,1)];
%gridParams.ACVH = ACVH; gridParams.bCVH = bCVH;
%gridParams.sparseGrid = makeGridND([min(X)' max(X)'],N);
%[gridParams.YIdx gridParams.XToBox, gridParams.numPointsPerBox, gridParams.boxEvalPoints] = makeGrid(gridParams.sparseGrid,[min(X) max(X)],ACVH,bCVH,N,M,dim,X);
%%
%params = createParams(X,m);
%[optParams logLike statistics] = newtonBFGSLInit(params,X,sampleWeights,1,gridParams);
%optParams = double(optParams);

minLogLike = 1000;
for i = 1:1
	params = createParams(X,m);
	logLike = zeros(2,1);
	bfgsInitC(X,sampleWeights,params,[min(X)' max(X)'],ACVH,bCVH,logLike);

%	if logLike(1) < minLogLike
%		fprintf('Choose run %d\n',i);
%		optParams = double(params);
%		minLogLike = logLike(1);
%	end
end
optParams = params;
aOpt = reshape(optParams(1:m*dim),[],dim); bOpt = optParams(m*dim+1:end);

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
	params = paramFitKernelDensity(X,sampleWeights,cvh);
end
toc(timeInit);


function params = createParams(X,n)
	[N dim] = size(X);
	lenP = n*(dim+1);
	a = rand(n,dim)*0.1; b = rand(n,1);
	params = [reshape(a,[],1); b];
end

end
