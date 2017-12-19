function [A b cvh] = calcCvxHullHyperplanes(X,cvh);

dim = size(X,2);

if dim == 1
	A = [1 -1]'; 
	b = [max(X) -min(X)]'; 
	[~,idxMax] = max(X);
	[~,idxMin] = min(X);
	cvh = [idxMin idxMax];
else
	mu = mean(X);

	if nargin == 1
		cvh = convhulln(X);
	end

	A = zeros(length(cvh),dim); b = zeros(length(cvh),1);
	for i = 1:length(cvh)
		B = X(cvh(i,1:dim-1),:)-X(cvh(i,[2:dim]),:);
		A(i,:) = null(B);
		% test orientation with a sample mean
		if A(i,:)*(X(cvh(i,1),:)-mu)' < 0
			A(i,:) = -A(i,:);
		end
		b(i) = A(i,:)*X(cvh(i,1),:)';
	end
end
