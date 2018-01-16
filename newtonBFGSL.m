function [params logLike statistics] = newtonBFGSL(params,X,sW,gamma,gridParams,options)
% *** TODO WRITE HELP ***
totTime = tic;
if nargin < 6
	options = struct();
end
options = setDefaults(options);

dim = size(X,2); lenP = length(params);
n = length(params)/(dim+1); N = length(X); M = length(gridParams.YIdx);
a = params(1:dim*n); a = reshape(a,[],dim); b = params(dim*n+1:end);
influence = zeros(n,1); statistics = struct();
evalFunc = zeros(M,1); evalFuncFloat = zeros(M,1,'single');

timing.evalGrid = 0;
timing.preCondGrad = 0;
timing.calcGradFast = 0;
timing.calcNewtonStep = 0;
timing.reduceHypers = 0;

% line search parameters
alpha = 10^-4; c2 = 0.9; beta = 0.1;
mode = 'normal';
timeA = struct(); timeB = struct();
tic; [gradA gradB TermA TermB] = calcGradFloat(single(X),single(gridParams.grid(1:dim)),a',b,gamma,gridParams.weight,single(gridParams.delta),influence,single(sW),gridParams.gridSize,gridParams.YIdx,gridParams.numPointsPerBox,single(gridParams.boxEvalPoints),gridParams.XToBox,gridParams.M,evalFuncFloat); grad = gradA + gradB; timing.evalGrid = timing.evalGrid + toc;
% the initial step is pure gradient descent
newtonStep = -grad;
statistics.initialIntegral = TermB;

% LBFGS variables
% begin the optimziation with floats, for higher precision we switch to double later
type = 'single';
m = min(40,ceil(n/5)); % number of previous gradients that are used to calculate the inverse Hessian
s_k = zeros(lenP,m);
y_k = zeros(lenP,m);
activeCol = 1;
sy = zeros(1,m); syInv = zeros(1,m);
updateList = 0; updateListInterval = 5;
numIter = 10000;
switchIter = -40;

for iter = 1:numIter
	timeOldStep = timing.evalGrid+timing.calcGradFast+timing.preCondGrad;
	numHypersHist(iter) = length(b); inactivePlanes = [];
	updateList = updateList-1;
	% remove inactive hyperplances
	if options.reduceHyperplanes && iter > 1 && length(b) > 1 && length(b) > options.minHyperplanes
		tic;
		inactivePlanes = find(influence < options.cutoff);
		% remove at most half the hyperplanes to prevent numerical instability
		if length(inactivePlanes) > length(b)/2
			randIdx = randperm(length(inactivePlanes));
			inactivePlanes = inactivePlanes(randIdx(1:round(length(b)/2)));
		end

		if length(inactivePlanes) > 0.01*length(b)
			idxRemove = inactivePlanes;
			for j = 2:dim+1
				idxRemove = [idxRemove inactivePlanes+n*(j-1)];
			end
			s_k(idxRemove,:) = []; y_k(idxRemove,:) = [];
			
			params(idxRemove) = []; a(inactivePlanes,:) = []; b(inactivePlanes) = []; lenP = length(b)*(dim+1); n = length(b);
			grad(idxRemove) = []; influence = zeros(length(b),1); newtonStep(idxRemove) = []; 
			if strcmp(mode,'fast')
				tic; [numEntries maxElements idxEntries elementList] = preCondGradAVX(single(X),single(gridParams.grid(1:dim)),a,b,gamma,gridParams.weight,single(gridParams.delta),gridParams.YIdx,gridParams.numPointsPerBox,single(gridParams.boxEvalPoints),a'); numEntries = [0 cumsum(numEntries)]; timing.preCondGrad = timing.preCondGrad + toc;
				
				if length(inactivePlanes) > 0.05*length(b);
					updateListInterval = round(updateListInterval/2);
				end
				updateList = updateListInterval;
			end
			% adapt m to reduced problem size
			if m > lenP/10
             	mOld = m;
                m = min(round(lenP/2),round(m/2));

                if (iter <= m)
                    vecCut = 1:m;
                else
                    vecCut = activeCol-m+1:activeCol;
                    vecCut(vecCut < 1) = vecCut(vecCut<1) + mOld;
                    activeCol = m;
                end
                s_k = s_k(:,vecCut); y_k = y_k(:,vecCut); sy = sy(vecCut); syInv = syInv(vecCut);
			end
		end
		timing.reduceHypers = timing.reduceHypers + toc;
	end

	% after convergence of hyperplanes switch mode
	if iter > 25 && (numHypersHist(end-25)-numHypersHist(end))/numHypersHist(end) < 0.05 && strcmp(mode,'normal') && length(b) > 500 && gamma >= 100 %&& numHypersHist(1)/numHypersHist(iter) > 5
        mode = 'fast';
        if options.verbose > 1
            fprintf('Change mode\n');
        end
		updateList = updateListInterval;
		tic; [numEntries maxElements idxEntries elementList] = preCondGradAVX(single(X),single(gridParams.grid(1:dim)),a,b,gamma,gridParams.weight,single(gridParams.delta),gridParams.YIdx,gridParams.numPointsPerBox,single(gridParams.boxEvalPoints),a'); numEntries = [0 cumsum(numEntries)]; timing.preCondGrad = timing.preCondGrad + toc;
	end

	lambdaSq = grad'*-newtonStep;
	if lambdaSq < 0 || lambdaSq > 1e5
		newtonStep = -grad;
		lambdaSq = norm(grad)^2;
		if options.verbose > 2
			if lambdaSq < 0
				fprintf('Negative step detected: lambdaSq = %.4f\n',lambdaSq);
			else
				fprintf('Numerical errors detected --> use gradient as newton step\n');
			end
		end
	end

	step = 1;
	% objective function value before the step
	TermAOld = TermA; TermBOld = TermB; funcVal = double(TermAOld + TermBOld); gradOld = grad;
	% new parameters
	paramsNew = params + step*newtonStep; aNew = paramsNew(1:dim*n); aNew = reshape(aNew,[],dim); bNew = paramsNew(dim*n+1:end);
	% funcVal after the step
	if strcmp(mode,'normal')
		if strcmp(type,'double')
			tic; [gradA gradB TermA TermB] = calcGrad(X,gridParams.grid(1:dim),aNew,bNew,gamma,gridParams.weight,gridParams.delta,influence,sW,gridParams.gridSize,gridParams.YIdx,gridParams.numPointsPerBox,gridParams.boxEvalPoints,gridParams.XToBox,gridParams.M,evalFunc);  timing.evalGrid = timing.evalGrid + toc; grad = gradA + gradB;
		else
			tic; [gradA gradB TermA TermB] = calcGradFloat(single(X),single(gridParams.grid(1:dim)),aNew',bNew,gamma,gridParams.weight,single(gridParams.delta),influence,single(sW),gridParams.gridSize,gridParams.YIdx,gridParams.numPointsPerBox,single(gridParams.boxEvalPoints),gridParams.XToBox,gridParams.M,evalFuncFloat); grad = gradA + gradB; timing.evalGrid = timing.evalGrid + toc;
		end
	else
		if strcmp(type,'double')
			tic; [grad TermA TermB] = calcGradFast(X,gridParams.grid(1:dim),double(aNew),double(bNew),gamma,gridParams.weight,gridParams.delta,influence,sW,gridParams.gridSize,gridParams.YIdx,numEntries,elementList,maxElements,idxEntries,evalFunc);  timing.calcGradFast = timing.calcGradFast + toc; 
		else
			tic; [grad TermA TermB] = calcGradFastFloat(single(X),single(gridParams.grid(1:dim)),single(aNew'),single(bNew),gamma,gridParams.weight,single(gridParams.delta),influence,single(sW),gridParams.gridSize,gridParams.YIdx,numEntries,elementList,maxElements,idxEntries); timing.calcGradFast = timing.calcGradFast + toc; grad = double(grad);
		end
		% update active hyperplane list
		if updateList <= 0
	%		fprintf('Update element list\n');
			tic; [numEntries maxElements idxEntries elementList] = preCondGradAVX(single(X),single(gridParams.grid(1:dim)),aNew,bNew,gamma,gridParams.weight,single(gridParams.delta),gridParams.YIdx,gridParams.numPointsPerBox,single(gridParams.boxEvalPoints),aNew'); numEntries = [0 cumsum(numEntries)]; timing.preCondGrad = timing.preCondGrad + toc;
			% check current error
			gradCheck = grad;
			if strcmp(type,'double')
				tic; [grad TermA TermB] = calcGradFast(X,gridParams.grid(1:dim),double(aNew),double(bNew),gamma,gridParams.weight,gridParams.delta,influence,sW,gridParams.gridSize,gridParams.YIdx,numEntries,elementList,maxElements,idxEntries,evalFunc);  timing.calcGradFast = timing.calcGradFast + toc;
			else
				tic; [grad TermA TermB] = calcGradFastFloat(single(X),single(gridParams.grid(1:dim)),single(aNew'),single(bNew),gamma,gridParams.weight,single(gridParams.delta),influence,single(sW),gridParams.gridSize,gridParams.YIdx,numEntries,elementList,maxElements,idxEntries);  grad = double(grad); timing.calcGradFast = timing.calcGradFast + toc;
			end
			% if the error is small, increase number of iterations until next update
			if norm(grad-gradCheck) < 10^-5
				updateListInterval = min(updateListInterval*2,100);
			else
				updateListInterval = round(updateListInterval/2);
			end
			updateList = updateListInterval;
		end
	end
	funcValStep = double(TermA + TermB);

	% backtracking line search with Wolfe conditions (see Nocedal Chapter 3)
	while isnan(funcValStep) || isinf(funcValStep) || funcValStep > funcVal - step*alpha*lambdaSq %|| grad'*newtonStep < c2*gradOld'*newtonStep
		if step < 10^-10
			break
		end
		step = beta*step;
		paramsNew = params + step*newtonStep; aNew = paramsNew(1:dim*n); aNew = reshape(aNew,[],dim); bNew = paramsNew(dim*n+1:end);

		if strcmp(mode,'normal')
			if strcmp(type,'double')
				  tic; [gradA gradB TermA TermB] = calcGrad(X,gridParams.grid(1:dim),aNew,bNew,gamma,gridParams.weight,gridParams.delta,influence,sW,gridParams.gridSize,gridParams.YIdx,gridParams.numPointsPerBox,gridParams.boxEvalPoints,gridParams.XToBox,gridParams.M,evalFunc);  timing.evalGrid = timing.evalGrid + toc; grad = gradA + gradB;
			 else
				tic; [gradA gradB TermA TermB] = calcGradFloat(single(X),single(gridParams.grid(1:dim)),aNew',bNew,gamma,gridParams.weight,single(gridParams.delta),influence,single(sW),gridParams.gridSize,gridParams.YIdx,gridParams.numPointsPerBox,single(gridParams.boxEvalPoints),gridParams.XToBox,gridParams.M,evalFuncFloat); grad = gradA + gradB; timing.evalGrid = timing.evalGrid + toc;
			end
		else
			if strcmp(type,'double')
				tic; [grad TermA TermB] = calcGradFast(X,gridParams.grid(1:dim),double(aNew),double(bNew),gamma,gridParams.weight,gridParams.delta,influence,sW,gridParams.gridSize,gridParams.YIdx,numEntries,elementList,maxElements,idxEntries,evalFunc);  timing.calcGradFast = timing.calcGradFast + toc; %grad = single(grad);
			else
				tic; [grad TermA TermB] = calcGradFastFloat(single(X),single(gridParams.grid(1:dim)),single(aNew'),single(bNew),gamma,gridParams.weight,single(gridParams.delta),influence,single(sW),gridParams.gridSize,gridParams.YIdx,numEntries,elementList,maxElements,idxEntries); timing.calcGradFast = timing.calcGradFast + toc;  grad = double(grad);
			end
		end
		funcValStep = double(TermA + TermB);
	end
	statistics.TermA(iter) = TermA; statistics.TermB(iter) = TermB;	stepHist(iter) = funcVal-funcValStep;

	if (funcVal-funcValStep == 0 && strcmp(type,'single'))
		if options.verbose > 1
			fprintf('Switch to double precision\n');
		end
		type = 'double';
		switchIter = iter;
	end

	% numerical errors during the opimtization
	if sum(isnan(grad)) > 0 || isinf(TermB) || isinf(TermA)
		fprintf('Numerical errors detected during the optimization: Return last correct result. Rerun the algorithm if necessary.');
		break
	end
	params = paramsNew; a = params(1:dim*n); a = reshape(a,[],dim); b = params(dim*n+1:end);

	% regular termination
	if abs(1-TermB) < options.intEps && mean(abs(stepHist(end))) < options.lambdaSqEps && iter > 10 && iter - switchIter > 50
		break
    end
	newtonStepOld = newtonStep;
	tic; newtonStep = calcNewtonStepC(s_k,y_k,sy,syInv,step,grad,gradOld,newtonStep,min([m,iter,length(b)]),activeCol-1,m); timing.calcNewtonStep = timing.calcNewtonStep + toc;

	activeCol = activeCol+1;
	if activeCol > m
		activeCol = 1;
	end
	if options.verbose > 1
		if ~mod(iter,5) || iter < 10
%			 fprintf('%d: %.5f (Primal: %.5f, Dual: %.5f) (%.4f, %.5f, %d) \t (lambdaSq: %.4e, t: %.2e, Step: %.4e) \t (Nodes per ms: %.2e)\n',iter,funcVal,TermA,statistics.dualFuncVal(iter),-TermAOld*N,TermBOld,length(b),lambdaSq,step,funcVal-funcValStep,M*length(b)/1000/(timing.evalGrid - timeOldStep));
			 fprintf('%d: %.5f (%.4f, %.5f, %d) \t (lambdaSq: %.4e, t: %.2e, Step: %.4e) \t (Nodes per ms: %.2e) %d \n',iter,funcVal,-TermAOld*N,TermBOld,length(b),lambdaSq,step,funcVal-funcValStep,(M+N)*length(b)/1000/(timing.evalGrid+timing.calcGradFast+timing.preCondGrad - timeOldStep),updateListInterval);
		end
	end
end
% output 
logLike = TermA*N; runTime = toc(totTime);
params = double(params);
% save some statistics
statistics.iterations = iter; statistics.influence = influence;
statistics.timings.preCondGrad = timing.preCondGrad; statistics.timings.calcGradFast = timing.calcGradFast; statistics.timings.evalGrid = timing.evalGrid;  statistics.timings.calcNewtonStep = timing.calcNewtonStep; statistics.timings.reduceHypers = timing.reduceHypers; statistics.timings.optimization = runTime; %statistics.timing = timing; 
statistics.timings.optimization = runTime;
statistics.numHypersHist = numHypersHist;
statistics.timeA = timeA; statistics.timeB = timeB;
%statistics.dualFuncVal = dualFuncVal;
% print final result for the optimization
if options.verbose > 0
	fprintf('Optimization with L-BFGS (CPU) finished: Iterations: %d, LambdaSq: %.4e, LogLike: %.4f, Integral: %.4e, Run time: %.2fs\n',iter,lambdaSq/2,TermA*N,abs(1-TermB),runTime);
	fprintf('%d planes remaining\n', length(b));
end

end

function options = setDefaults(options)

if ~isfield(options,'verbose')
	options.verbose = 0;
end

if ~isfield(options,'reduceHyperplanes')
	options.reduceHyperplanes = 1;
end

if ~isfield(options,'lambdaSqEps')
    options.lambdaSqEps = 10^-6;
end

if ~isfield(options,'intEps')
    options.intEps = 10^-3;
end

if ~isfield(options,'cutoff')
	options.cutoff = 0.1;
end

end
