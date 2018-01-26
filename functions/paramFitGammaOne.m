function params = paramFitGammaOne(X,sampleWeights,ACVH,bCVH,cvh) 

[n dim] = size(X);
m = 10*dim;

minLogLike = 1000;
for i = 1:1
	a = rand(m*dim,1)*0.1; b = rand(m,1);
	params = [a; b];
	logLike = zeros(2,1);
	bfgsInitC(X,sampleWeights,params,[min(X)' max(X)'],ACVH,bCVH,logLike);

	if logLike(1) < minLogLike
		%fprintf('Choose run %d\n',i);
		optParams = double(params);
		minLogLike = logLike(1);
	end
end
aOpt = reshape(optParams(1:m*dim),[],dim); bOpt = optParams(m*dim+1:end);

yT = -log(sum(exp(aOpt*X' + repmat(bOpt,1,length(X)))))';

idxCVH = unique(cvh);
T = int32(convhulln([X yT; X(idxCVH,:) repmat(min(yT(idxCVH))-1,length(idxCVH),1)]));
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


% detect if all hyerplanes where intitialized to the same parameters; happens for small sample sizes --> very slow convergence in the final optimization
if mean(var(aOpt(randperm(length(bOpt),min(100,length(bOpt))),:))) < 10^-4 || length(params) == dim+1
	fprintf('#### Bad initialization due to small sample size, switch to kernel kensity based initialization ####\n');
	params = paramFitKernelDensity(X,sampleWeights,cvh);
else
	params = [aOpt bOpt];
end

end
