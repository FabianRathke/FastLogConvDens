function [samples samplesEval] = sampleLCDM(gridParams,n)

d = size(gridParams.X,2);

% use the rejection sampling method described in Cule et al.
probs = gridParams.Ad.*gridParams.Gd;
probs = probs/sum(probs);
lenT = length(gridParams.Ad);

samplesSimplex = int32(datasample(1:lenT,n,'Weights',probs));

%samples = zeros(n,d);
%samplesSimplex = datasample(1:lenT,n,'Weights',probs);
%for i = 1:n 
%	while (sum(samples(i,:) == 0))
%		w = exprnd(1,1,d+1);
%		w = w/sum(w);
%		y = gridParams.yT(gridParams.T(samplesSimplex(i),:));
%		fw = exp(y*w');
%		maxfx =  max(exp(y));
%		u = rand;
%		if (u < fw/maxfx) 
%			samples(i,:) = w*gridParams.X(gridParams.T(samplesSimplex(i),:),:);
%
%			samplesEval(i) = fw;
%		end
%	end
%end


% first draw a number simplices
samples = zeros(n,d);

toSample = 1:n;
numSamples = length(toSample);
% sample until we obtained all samples
while numSamples > 0
	% sample points from the unit-simplex
	w = exprnd(1,numSamples,d+1);
	w = w./repmat(sum(w,2),1,d+1);
	% evaluate points with density
	fw = exp(sum(w.*gridParams.yT(gridParams.T(samplesSimplex(toSample),:)),2));
	maxFx = max(exp(gridParams.yT(gridParams.T(samplesSimplex(toSample),:))),[],2);
	% rejection sampling
	accept = rand(numSamples,1) < fw./maxFx;

	for i = 1:d
		samples(toSample(accept),i) = sum(reshape(reshape(w(accept,:)',1,[]).*gridParams.X(reshape(gridParams.T(samplesSimplex(toSample(accept)),:)',1,[]),i)',d+1,[]));
	end
	samplesEval(toSample(accept)) = fw(accept);
	toSample(accept) = [];
	numSamples = length(toSample);

	if sum(samples(:,1)~=0) > n
		break;
	end
end

nonZero = find(samples(:,1)~=0);
samplesSimplex = samplesSimplex(nonZero(1:n));
samples = samples(nonZero(1:n),:);
samplesEval = samplesEval(nonZero(1:n));
