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
XSave = X;
optOptions = struct('verbose',options.verbose,'reduceHyperplanes',1,'minHyperplanes',0,'method',@newtonBFGSLCTest,'cutoff',10^-2,'lambdaSqEps',10^-7,'intEps',10^-4,'returnGrid',1,'sampleWeights',ones(1,lenX));
[gridParams X optOptions] = obtainGrid(X,optOptions);
posterior = posterior(gridParams.idxSet,:);
options.classID = options.classID(gridParams.idxSet);
gamma = 1000;
minLogLike = -inf;
optOptions.correctIntegral = 1;

% initialize both densities; after that use previous parameters as initializations
for j = 1:K
	optOptions.sampleWeights = posterior(:,j)./sum(posterior(:,j));
	[a b ll statistics] = lcdFast(X,gamma,optOptions);
	% E-Step
	% evaluate marginal densities p(x|z,\beta)
%	evalInner = gamma*(a*X' + repmat(b,1,lenX));
%    densEst(:,j) = sum(exp(evalInner-repmat(max(evalInner),length(b),1))).^(1/gamma).*exp(-max(evalInner)/gamma);	
	densEst(:,j) = exp(-max(a*X' + repmat(b,1,lenX)));
	tau(j) = N_k(j)/lenX;
	params{j} = [reshape(a,[],1); b];
end

for iter = 1:1000
  	 % evaluate the log likelihood p(X|\beta)
  	logLike(iter) = sum(log(sum(repmat(tau,lenX,1).*densEst,2)));
	if isfield(options,'classID')
		[~,IDXLog] = max((repmat(tau,lenX,1).*densEst)');
		fprintf('# Missclassified: %d\n',min(lenX-sum((IDXLog'==1)==(options.classID==0)),lenX-sum((IDXLog'==2)==(options.classID==0))));
	end
	fprintf('%d: Log-Likelihood %.4f\n',iter,logLike(iter));

	if iter > 5
		likeDelta = abs((logLike(end-2)-logLike(end))/(logLike(end)));
		fprintf('%.3e\n',likeDelta);
		if likeDelta < 2*10^-5 || iter == 50
	   	   	break
		end
	end

	% update posterior probabilities p(z|x,\beta)
    for j = 1:K
        posterior(:,j) = tau(j)*densEst(:,j)./sum(repmat(tau,lenX,1).*densEst,2);
        N_k(j) = sum(posterior(:,j));
    end

   	% M-Step (update density parameters and mixing property tau)
	for j = 1:K
		optOptions.sampleWeights = posterior(:,j)./sum(posterior(:,j));
		optOptions.sampleWeights(optOptions.sampleWeights < 10^-8/lenX) = 0;
		fprintf('%d filtered\n',sum(optOptions.sampleWeights==0));
%		XTmp = X;
%		XTmp(optOptions.sampleWeights==0),:) = [];
		% use old params as initialization
		[params{j} ll statistics] = optOptions.method(params{j},X,gamma,gridParams,optOptions);
		numHypers = length(params{j})/(dim+1);
		a = reshape(params{j}(1:dim*numHypers),[],dim); b = params{j}(dim*numHypers+1:end);
		if options.fast
			evalInner = gamma*(a*X' + repmat(b,1,lenX));
	        densEst(:,j) = sum(exp(evalInner-repmat(max(evalInner),length(b),1))).^(1/gamma).*exp(-max(evalInner)/gamma);
		else
			[statistics a b ll T yT] = correctIntegral(X,zeros(1,dim),optOptions.sampleWeights,a,b,struct(),optOptions,gridParams.cvh);
			params{j} = [reshape(a,[],1); b];
		    % evaluate density for X
	%		densEst(:,j) = exp(-max(a*X' + repmat(b,1,lenX)));
			densEst(:,j) = exp(yT);
		end
		tau(j) = N_k(j)/lenX;
	end
end

if options.fast || optOptions.correctIntegral == 0
	optOptions.correctIntegral = 1;
	for j = 1:K
		sW = posterior(:,j)./sum(posterior(:,j));
		sW(sW < 10^-8/lenX) = 0;
		numHypers = length(params{j})/(dim+1);
		aOpt = reshape(params{j}(1:dim*numHypers),[],dim); bOpt = params{j}(dim*numHypers+1:end);
		[statistics a b ll] = correctIntegral(X,zeros(1,dim),sW,aOpt,bOpt,struct(),optOptions,gridParams.cvh);
		params{j} = [reshape(a,[],1); b];
		densEst(:,j) = exp(-max(a*XSave' + repmat(b,1,lenX)));
	end
	logLike(iter+1) = sum(log(sum(repmat(tau,lenX,1).*densEst,2)));
	fprintf('Final Log-likelihood: %.4f\n',logLike(iter+1));
else
	for j=1:K
		numHypers = length(params{j})/(dim+1);
		a = reshape(params{j}(1:dim*numHypers),[],dim); b = params{j}(dim*numHypers+1:end);
		evalInner = gamma*(a*XSave' + repmat(b,1,lenX));
		densEst(:,j) = sum(exp(evalInner-repmat(max(evalInner),length(b),1))).^(1/gamma).*exp(-max(evalInner)/gamma);
	end
end

%logLike = minLogLike;
%params = minParams;
%densEst = minDensEst;

paramsReturn.dimData = dim;
paramsReturn.logLike = logLike;
paramsReturn.timeReq = toc(timing);
paramsReturn.tau = tau;
paramsReturn.params = params;
paramsReturn.gridParams = gridParams;
