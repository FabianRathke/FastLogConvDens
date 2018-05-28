if exist('gridParams')
	ACVH = gridParams.ACVH;
	bCVH = gridParams.bCVH;
end
box = double([min(X)' max(X)']);
if exist('sampleWeights')
	sW = sampleWeights;
end
if exist('paramsKernel')
	paramsA = params;
	paramsB = paramsKernel;
	save([getenv('WORKHOME'), 'MyCode/LogConcave/code/newtonInit.mat'],'X','sW','ACVH','bCVH','box','paramsA','paramsB');
else
	save([getenv('WORKHOME'), 'MyCode/LogConcave/code/newtonInit.mat'],'X','sW','ACVH','bCVH','box','params');
end
%params = reshape(params,[],size(X,2)+1);
%randIdx = randperm(size(params,1));
%params = params(randIdx,:);
%params = reshape(params,[],1);

