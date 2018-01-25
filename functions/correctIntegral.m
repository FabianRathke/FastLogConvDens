function [aOpt bOpt T yT Ad Gd aOptOld bOptOld] = correctIntegral(X,mu,aOpt,bOpt,cvh)

[n dim] = size(X);

% move hyperplane parameters to the actual mean mu
% create random set of points and calculate the corresponding y points == -log(f(x))
XTmp = rand(length(bOpt)*(dim+1),dim);
yTmp = sum(repmat(-aOpt,(dim+1),1).*XTmp,2) - repmat(bOpt,dim+1,1);
% add mean
XTmp = XTmp + repmat(mu,length(bOpt)*(dim+1),1);
T = int32(repmat([1:length(bOpt):length(bOpt)*(dim+1)],length(bOpt),1) + repmat((0:length(bOpt)-1)',1,dim+1));
% calc new params
[aOptNew bOptNew] = calcExactIntegral(XTmp',yTmp,T'-1,dim,0); aOptNew = aOptNew';

% save sparse hyperplane parameters
aOptOld = aOptNew; bOptOld = bOptNew;
if (length(bOpt)==1)
	yT = (-aOpt*X'-repmat(bOpt,1,length(X)));
else
	yT = min(-aOpt*X'-repmat(bOpt,1,length(X)));
end
idxCVH = unique(cvh);
T = int32(convhulln([X yT'; X(idxCVH,:) repmat(min(yT)-1,length(idxCVH),1)])); 
T(max(T,[],2)>length(yT),:) = [];

% add back mean
X = X + repmat(mu,n,1);

[aOptNew bOptNew integral changeB Ad Gd] = calcExactIntegral(X',yT,T'-1,dim,1); 
aOptNew = aOptNew';
if norm(max(aOptNew*X(1:min(size(X,1),10),:)' + repmat(bOptNew,1,min(length(yT),10)),[],1) + yT(1:min(10,length(yT)))) > 10^-6
    warning('Potential numerical problems when calculating the final set of hyperplanes --> Recommended to run the optimization again');
else
	aOpt = aOptNew; bOpt = bOptNew;
end
