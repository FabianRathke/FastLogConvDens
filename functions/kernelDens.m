function y = kernelDens(X,sampleWeights)
	[n d] = size(X);
	h = std(X)*n^(-1/(d+4));
	y = kernelDensC(X',sampleWeights,h);
end
