function [N M grid weight gridSize] = setGridDensity(box,dim,sparse,optOptions)

% use default densitities if nothing was specified by the user
if sparse
    if ~isfield(optOptions,'NInit') || ~isfield(optOptions,'MInit')
        [N M] = getMN(dim,sparse);
    else
        N = optOptions.NInit; M = optOptions.MInit;
    end
else
    if ~isfield(optOptions,'N') || ~isfield(optOptions,'M')
        [N M] = getMN(dim,sparse);
    else
        N = optOptions.N; M = optOptions.M;
    end
end

gridSize = N*M;
% grid for midpoint rule
for i = 1:dim
    interval = linspace(box(i,1),box(i,2),gridSize+1);
    grid(i,:) = interval(1:end-1)+(interval(2)-interval(1))/2;
end

weight = prod(grid(:,2)-grid(:,1));
end
