[gridParams X optOptions] = obtainGrid(X,optOptions);
a = params(:,1:dim); b = params(:,end); influence = zeros(length(params),1);
evalFunc = zeros(length(gridParams.YIdx),1,'single');
[gradA gradB TermA TermB] = calcGradFloat(single(X),single(gridParams.grid(1:dim)),a',b,gamma,gridParams.weight,single(gridParams.delta),influence,single(sW),gridParams.gridSize,gridParams.YIdx,gridParams.numPointsPerBox,single(gridParams.boxEvalPoints),gridParams.XToBox,gridParams.M,evalFunc);

a = paramsKernel(:,1:dim); b = paramsKernel(:,end); influence = zeros(length(paramsKernel),1);
[gradA gradB TermAKernel TermBKernel] = calcGradFloat(single(X),single(gridParams.grid(1:dim)),a',b,gamma,gridParams.weight,single(gridParams.delta),influence,single(sW),gridParams.gridSize,gridParams.YIdx,gridParams.numPointsPerBox,single(gridParams.boxEvalPoints),gridParams.XToBox,gridParams.M,evalFunc);

if optOptions.verbose > 1
	fprintf('############ %.4f, %.4f, %.4f ############\n',(TermA+TermB) - (TermAKernel+TermBKernel),(TermA+TermB),(TermAKernel+TermBKernel));
end

if (TermA+TermB) > (TermAKernel+TermBKernel)
	params = paramsKernel;
	if optOptions.verbose > 1
		fprintf('Use kernel initialization\n');
	end
	initSelect = 'kernel';
else
	if optOptions.verbose > 1
		fprintf('Use gamma intialization\n');
	end
	initSelect = 'gamma';
end
