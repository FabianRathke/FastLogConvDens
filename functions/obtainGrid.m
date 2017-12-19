function [gridParams X optOptions] = obtainGrid(X,optOptions,gridInit);

if nargin < 3
	gridInit = 0;
end

[n dim] = size(X);
[gridParams.ACVH gridParams.bCVH gridParams.cvh] = calcCvxHullHyperplanes(X);

% grid denotes the coordinates of the integration grid (gridParams.YIdx refers to this grid; with C-indexing starting with 0)
[N M gridParams.grid gridParams.weight gridParams.gridSize] = setGridDensity(X,dim,gridInit,optOptions);
gridParams.N = N; gridParams.M = M;
gridParams.delta = [gridParams.grid(:,2)-gridParams.grid(:,1)];

% sparseGrid holds left lower points of all boxes that encompass the sparse subgrid; gridParams.boxIDs refers to this points; with C-indexing starting with 0)
% each box contains at most M^dim grid points and there are at most N^dim boxes
gridParams.sparseGrid = makeGridND([min(X)' max(X)'],N,'trapezoid');
gridParams.sparseDelta = (gridParams.sparseGrid(:,end)-gridParams.sparseGrid(:,1))/gridParams.N;
[gridParams.YIdx gridParams.gridToBox gridParams.XToBox, gridParams.numPointsPerBox, gridParams.boxEvalPoints, gridParams.YIdxSub gridParams.subGrid gridParams.subGridIdx gridParams.boxIDs gridParams.XToBoxOuter] = makeGrid(gridParams.sparseGrid,[min(X) max(X)],gridParams.ACVH,gridParams.bCVH,N,M,dim,X,gridParams.sparseDelta);
% sort X according to the corresponding boxes
[XToBoxSorted,idxSet] = sort(gridParams.XToBox);
X = X(idxSet,:);
gridParams.idxSet = idxSet;
optOptions.sampleWeights = optOptions.sampleWeights(idxSet);
%gridParams.cvh = idxSet(gridParams.cvh);
[A B C] = unique([idxSet 1:length(idxSet)]);
gridParams.cvh = B(gridParams.cvh);
gridParams.XToBox = XToBoxSorted;
gridParams.XToBoxOuter = gridParams.XToBoxOuter(idxSet);
end

