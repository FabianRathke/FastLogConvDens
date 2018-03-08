function [densEst paramsReturn logLike] = lcdFastEM(X,posterior,options)

timing = tic;

% find duplicates in X
%[X,IA,IC] = unique(X,'rows');
%posterior = posterior(IA,:);
%if isfield(options,'classID')
%	options.classID = options.classID(IA);
%end
%w = hist(IC,length(IA))'/length(IA);

N_k = sum(posterior);
K = length(N_k);
[lenX dim] = size(X);

% options for the optimization
optOptions = struct('verbose',options.verbose-1,'cutoff',10^-1,'lambdaSqEps',10^-8,'intEps',10^-4);
[gridParams.ACVH gridParams.bCVH gridParams.cvh] = calcCvxHullHyperplanes(X);
gamma = 1000;
minLogLike = -inf;
maxIter = 50;

% initialize both densities; after that use previous parameters as initializations
for j = 1:K
	optOptions.sampleWeights = posterior(:,j)./sum(posterior(:,j));
	[a b] = lcdFast(X,gamma,optOptions);
	densEst(:,j) = exp(-max(a*X' + repmat(b,1,lenX)));
	tau(j) = N_k(j)/lenX;
	params{j} = [reshape(a,[],1); b];
end

for iter = 1:maxIter
  	 % evaluate the log likelihood p(X|\beta)
  	logLike(iter) = sum(log(densEst*tau'));
	if isfield(options,'classID')
		[~,IDXLog] = max((repmat(tau,lenX,1).*densEst)');
		if options.verbose > 0
			%fprintf('# Missclassified: %d\n',min(lenX-sum((IDXLog'==1)==(options.classID==0)),lenX-sum((IDXLog'==2)==(options.classID==0))));
		end
	end
	if options.verbose > 0
		fprintf('%d: Log-Likelihood %.4f\n',iter,logLike(iter));
	end

	% check for convergence
	if iter > 5
		likeDelta = abs((logLike(end-2)-logLike(end))/(logLike(end)));
		if likeDelta < 10^-5
	   	   	break
		end
	end

	% E-Step: updates posterior probabilities p(z|x,\beta)
    for j = 1:K
        posterior(:,j) = tau(j)*densEst(:,j)./(densEst*tau');
    end

   	% M-Step: updates density parameters and mixing property tau
	for j = 1:K
		optOptions.sampleWeights = posterior(:,j)./sum(posterior(:,j));
		%optOptions.sampleWeights(optOptions.sampleWeights < 10^-8/lenX) = 0;
		filter = optOptions.sampleWeights > 1e-8/lenX;
		[gridParams.ACVH gridParams.bCVH gridParams.cvh] = calcCvxHullHyperplanes(X(filter,:));
		% use old params as initialization
		params{j} = bfgsFullC(X(filter,:),optOptions.sampleWeights(filter),params{j},[],[min(X(filter,:))' max(X(filter,:))'],gridParams.ACVH,gridParams.bCVH,optOptions.verbose,optOptions.intEps, optOptions.lambdaSqEps, optOptions.cutoff);

		numHypers = length(params{j})/(dim+1);
		a = reshape(params{j}(1:dim*numHypers),[],dim); b = params{j}(dim*numHypers+1:end);
		[a b T yT] = correctIntegral(X(filter,:),zeros(1,dim),a,b,gridParams.cvh);
		params{j} = [reshape(a,[],1); b];
		% evaluate density for X
		densEst(filter,j) = exp(yT);
		densEst(optOptions.sampleWeights < 1e-8/lenX,j) = -inf;
		tau(j) = sum(posterior(:,j))/lenX;
	end
end

paramsReturn.dimData = dim;
paramsReturn.logLike = logLike;
paramsReturn.timeReq = toc(timing);
paramsReturn.tau = tau;
paramsReturn.params = params;
