function y = kernelDens(X,sampleWeights,gridParams)
	[n d] = size(X);
	h = std(X)*n^(-1/(d+4));
	hInv = 1./h.^2;

	if n > 10*length(unique(gridParams.XToBoxOuter))
		% sort X according to which box it belongs to
		[XToBoxOuter,idxSet] = sort(gridParams.XToBoxOuter);
		X = X(idxSet,:);

		notAssigned = XToBoxOuter==intmax('uint16');
		X(notAssigned,:) = [];
		XToBoxOuter(notAssigned) = [];

		% C implementation
		boxList = unique(XToBoxOuter);
		boxIDs = gridParams.boxIDs(boxList+1);
		sparseGrid = gridParams.sparseGrid(:,boxIDs+1);
		idxStart = uint32([find(int16(XToBoxOuter(1:end))-[-1 int16(XToBoxOuter(1:end-1))])]); % first appearance of some X for each box
		numPoints = [idxStart(2:end) size(X,1)+1] - idxStart; % numbers of data points in each box
		idxStart = idxStart-1; % C indexing
		y = kernelDensC(X',sampleWeights,h,sparseGrid,gridParams.sparseDelta,idxStart,numPoints);
		y(notAssigned) = min(y);
		% redo the old ordering
		y(idxSet) = y;
	else
		y = kernelDensC(X',sampleWeights,h,[],[],unique(gridParams.XToBoxOuter),[]);
	end
	
%	[n d] = size(X);
	% matlab code
%	y = (sampleWeights'*squeeze(prod(exp((-(repmat(X,[1,1,n])-shiftdim(repmat(X',[1,1,n]),2)).^2)*0.5.*repmat(hInv,[n,1,n]))./(repmat(h*sqrt(2*pi),[n,1,n])),2))')';
    % old for-loop code
%	for i = 1:n
 %       x = X(i,:);
  %      yTest(i) = sampleWeights(~notAssigned)'*(prod(exp((-(repmat(x,n,1)-X).^2)*0.5./repmat(h,n,1).^2)./(repmat(h,n,1)*sqrt(2*pi)),2));
  %  end
end

