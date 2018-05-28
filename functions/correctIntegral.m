function [aOpt bOpt T yT Ad Gd aOptSparse bOptSparse] = correctIntegral(X,mu,aOpt,bOpt,cvh)

[n dim] = size(X);

% move hyperplane parameters to the actual mean mu
% create random set of points and calculate the corresponding y points == -log(f(x))
XTmp = rand(length(bOpt)*(dim+1),dim);
yTmp = sum(repmat(-aOpt,(dim+1),1).*XTmp,2) - repmat(bOpt,dim+1,1);
% add mean
XTmp = XTmp + repmat(mu,length(bOpt)*(dim+1),1);
T = int32(reshape(1:length(bOpt)*(dim+1),[],dim+1));
% calc new params
[aOptNew bOptNew] = calcExactIntegral(XTmp',yTmp,T'-1,dim,0); aOptNew = aOptNew';

% save sparse hyperplane parameters
aOptSparse = aOptNew; bOptSparse = bOptNew;

idxCVH = unique(cvh);
% for 1-D we can evaluate the sparse integral directly
if dim == 1
	[aOpt I] = sort(aOpt);
	bOpt = bOpt(I);
	
	X_ = [max(X); (bOpt(2:end)-bOpt(1:end-1))./(aOpt(1:end-1)-aOpt(2:end)); min(X)];
	X_ = sort(X_);
	X_(X_ > max(X)) = [];
	X_(X_ < min(X)) = [];
	X_save = X;
	X = X_;
	yT = min(-aOpt*X' - repmat(bOpt,1,length(X)));
	T = int32(convhulln([X yT'; X repmat(min(yT)-1,length(X),1)])); 
else
	if (length(bOpt)==1)
		yT = (-aOpt*X'-repmat(bOpt,1,length(X)));
	else
		yT = min(-aOpt*X'-repmat(bOpt,1,length(X)));
	end
	T = int32(convhulln([X yT'; X(idxCVH,:) repmat(min(yT)-1,length(idxCVH),1)])); 
end

T(max(T,[],2)>length(yT),:) = [];

% add back mean
X = X + repmat(mu,size(X,1),1);

[aOptNew bOptNew integral changeB Ad Gd] = calcExactIntegral(X',yT,T'-1,dim,1); 
aOptNew = aOptNew';
if norm(max(aOptNew*X(1:min(size(X,1),10),:)' + repmat(bOptNew,1,min(length(yT),10)),[],1) + yT(1:min(10,length(yT)))) > 10^-6
    warning('Potential numerical problems when calculating the final set of hyperplanes --> Recommended to run the optimization again');
else
	aOpt = aOptNew; bOpt = bOptNew;
end

if dim == 1
	X = X_save + repmat(mu, size(X_save,1),1);
	yT = min(-aOpt*X' - repmat(bOpt,1,length(X)));
end
