writeMatFile
n = length(params)/(dim+1); N = length(X); M = length(gridParams.YIdx);
a = params(1:dim*n); a = reshape(a,[],dim); b = params(dim*n+1:end);

% newtonBFGSL
[gradA gradB TermA TermB] = calcGradFloat(single(X),single(gridParams.grid(1:dim)),single(a)',single(b),gamma,gridParams.weight,single(gridParams.delta),influence,single(sW),gridParams.gridSize,gridParams.YIdx,gridParams.numPointsPerBox,single(gridParams.boxEvalPoints),gridParams.XToBox,gridParams.M,evalFuncFloat); grad = double(gradA + gradB);

% newtonBFGSLInit
[gradA gradB TermA TermB] = calcGradFloatAVX(single(X),single(gridParams.grid(1:dim)),single(a)',single(b),gamma,gridParams.weight,single(gridParams.delta),influence,single(sW),gridParams.gridSize,gridParams.YIdx); grad = double(gradA+gradB);
