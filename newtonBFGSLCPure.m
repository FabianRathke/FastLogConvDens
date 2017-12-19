function [params logLike statistics] = newtonBFGSLCPure(params,X,gamma,gridParams,options)
% *** TODO WRITE HELP ***
totTime = tic;
if nargin < 5
	options = struct();
end
options = setDefaults(options);

if ~isfield(options,'sampleWeights')
	sW = ones(length(X),1);
else
	sW = options.sampleWeights;
end
sW = sW'/sum(sW);

dim = size(X,2); lenP = length(params);
n = length(params)/(dim+1); N = length(X); M = length(gridParams.YIdx);
a = params(1:dim*n); a = reshape(a,[],dim); b = params(dim*n+1:end);
influence = zeros(n,1); statistics = struct();
evalFunc = zeros(M,1);

timing.evalGrid = 0;
timing.calcNewtonStep = 0;
timing.reduceHypers = 0;

% line search parameters
alpha = 10^-4; c2 = 0.9; beta = 0.1;
tic; [gradA gradB TermA TermB] = calcGrad(X,gridParams.grid(1:dim),a,b,gamma,gridParams.weight,gridParams.delta,influence,sW,gridParams.gridSize,gridParams.YIdx,gridParams.numPointsPerBox,gridParams.boxEvalPoints,gridParams.XToBox,gridParams.M,evalFunc);  timing.evalGrid = timing.evalGrid + toc; grad = gradA + gradB;

%tic; [gradFull TermAFull TermBFull] = calcGradFull(X,gridParams.grid(1:dim),a,b,gamma,gridParams.weight,gridParams.delta,influence,sW,gridParams.gridSize,gridParams.YIdx);  timing.evalGrid = timing.evalGrid + toc;
% the initial step is pure gradient descent
newtonStep = -grad;
statistics.initialIntegral = TermB;

% LBFGS variables
m = 40; % number of previous gradients that are used to calculate the inverse Hessian
s_k = zeros(lenP,m);
y_k = zeros(lenP,m);
sy = zeros(1,m); syInv = zeros(1,m);
updateList = 0; updateListInterval = 5;

for iter = 1:10000
	numHypersHist(iter) = length(b); inactivePlanes = [];

	% remove inactive hyperplances
	if options.reduceHyperplanes && iter > 1
		tic;
		inactivePlanes = find(influence < options.cutoff);

		if length(inactivePlanes) > 0.01*length(b)
			idxRemove = inactivePlanes;
			for j = 2:dim+1
				idxRemove = [idxRemove inactivePlanes+n*(j-1)];
			end
			s_k(idxRemove,:) = []; y_k(idxRemove,:) = [];
			
			params(idxRemove) = []; a(inactivePlanes,:) = []; b(inactivePlanes) = []; lenP = length(b)*(dim+1); n = length(b);
			grad(idxRemove) = []; influence = zeros(length(b),1); newtonStep(idxRemove) = [];
		end
     	if m > lenP/10
        	m = round(lenP/10);
            s_k = s_k(:,1:m); y_k = y_k(:,1:m); sy = sy(1:m); syInv = syInv(1:m);
        end
		timing.reduceHypers = timing.reduceHypers + toc;
	end


	lambdaSq = grad'*-newtonStep;
	if lambdaSq < 0
		newtonStep = -grad;
		lambdaSq = norm(grad)^2;
		if options.verbose > 0
			fprintf('Negative step detected: lambdaSq = %.4f\n',lambdaSq);
		end
	end

	if lambdaSq > 10^5
		newtonStep = -grad;
		lambdaSq = norm(grad)^2;
		if options.verbose > 0
			fprintf('Numerical errors detected --> use gradient as newton step\n');
		end
	end

	step = 1;
	% objective function value before the step
	TermAOld = TermA; TermBOld = TermB; funcVal = TermAOld + TermBOld; gradOld = grad;
	% new parameters
	paramsNew = params + step*newtonStep; aNew = paramsNew(1:dim*n); aNew = reshape(aNew,[],dim); bNew = paramsNew(dim*n+1:end);
	% funcVal after the step
	tic; [gradA gradB TermA TermB] = calcGrad(X,gridParams.grid(1:dim),aNew,bNew,gamma,gridParams.weight,gridParams.delta,influence,sW,gridParams.gridSize,gridParams.YIdx,gridParams.numPointsPerBox,gridParams.boxEvalPoints,gridParams.XToBox,gridParams.M,evalFunc);  timing.evalGrid = timing.evalGrid + toc; grad = gradA + gradB;
	funcValStep = TermA + TermB;

	% backtracking line search with Wolfe conditions (see Nocedal Chapter 3)
	while isnan(funcValStep) || isinf(funcValStep) || funcValStep > funcVal - step*alpha*lambdaSq %|| grad'*newtonStep < c2*gradOld'*newtonStep
		if step < 10^-9
			break
		end
		step = beta*step;
		paramsNew = params + step*newtonStep; aNew = paramsNew(1:dim*n); aNew = reshape(aNew,[],dim); bNew = paramsNew(dim*n+1:end);

		tic; [gradA gradB TermA TermB] = calcGrad(X,gridParams.grid(1:dim),aNew,bNew,gamma,gridParams.weight,gridParams.delta,influence,sW,gridParams.gridSize,gridParams.YIdx,gridParams.numPointsPerBox,gridParams.boxEvalPoints,gridParams.XToBox,gridParams.M,evalFunc);  timing.evalGrid = timing.evalGrid + toc; grad = gradA + gradB;
		funcValStep = TermA + TermB;
	end
	statistics.TermA(iter) = TermA; statistics.TermB(iter) = TermB;	stepHist(iter) = funcVal-funcValStep;
	dualFuncVal = sum(-evalFunc.*log(evalFunc))*gridParams.weight;
	statistics.dualFuncVal(iter) = dualFuncVal;	
	% numerical errors during the opimtization
	if sum(isnan(grad)) > 0 || isinf(TermB) || isinf(TermA)
		error('Numerical errors detected during the optimization: Rerun the algorithm.');
		break
	end
	params = paramsNew; a = params(1:dim*n); a = reshape(a,[],dim); b = params(dim*n+1:end);

	% regular termination
	if abs(1-TermB) < options.intEps && mean(abs(stepHist(max(end-10,1):end))) < options.lambdaSqEps && iter > 10
		break
    end

	tic;
	% move and delete entries
	s_k(:,2:end) = s_k(:,1:end-1); y_k(:,2:end) = y_k(:,1:end-1); sy(2:end) = sy(1:end-1); syInv(2:end) = syInv(1:end-1);

	% save new ones
	s_k(:,1) = step*newtonStep; 
    gammaBFGS = grad - gradOld;
	% Updates according to Li et al.
%	t = 1 + max([-gammaBFGS'*s_k(1,:)'/norm(s_k(1,:)).^2 0]);
 %  	y_k(1,:) = gammaBFGS + t*norm(gradOld)*s_k(1,:)';
	% Updates according to Wei et al.
%	t = (2*(funcVal-funcValStep) + (grad+gradOld)'*s_k(1,:)')/norm(s_k(1,:)).^2;
%	y_k(1,:) = gammaBFGS + t*s_k(1,:)';
	% Updates from Li but modified --> now the statement is true
	t = norm(gradOld) + max([-gammaBFGS'*s_k(:,1)/norm(s_k(:,1)).^2 0]);
 	y_k(:,1) = gammaBFGS' + t'*s_k(:,1)';

	sy(1) = s_k(:,1)'*y_k(:,1); syInv(1) = 1/sy(1);
	
	% choose H0
	gamma_k = sy(1)/(y_k(:,1)'*y_k(:,1));
	H0 =  gamma_k;

	% first for-loop
	q = grad;
	for l = 1:min([m,iter,length(b)]);
		alphaBFGS(l) = s_k(:,l)'*q*syInv(l);
		q = q - alphaBFGS(l)*y_k(:,l);
	end

	r = H0.*q;
	% second for-loop
	for l = min([m,iter,length(b)]):-1:1
		betaBFGS = y_k(:,l)'*r*syInv(l);
		r = r + s_k(:,l)*(alphaBFGS(l)-betaBFGS);
	end
	newtonStep = -r;
	timing.calcNewtonStep = timing.calcNewtonStep + toc;

	
	% print some information
	if options.verbose > 1
		if ~mod(iter,5)
			fprintf('%d: %.5f (Dual: %.5f, Primal: %.5f) (%.4f, %.5f, %d) \t (lambdaSq: %.4e, t: %.2e, Step: %.4e)\n',iter,funcVal,dualFuncVal,TermA,-TermAOld*N,TermBOld,length(b),lambdaSq,step,funcVal-funcValStep);
		end
	end
end

% final pruning of hyperplanes
%inactivePlanes = find(influence<(M/length(b)*10^-3))';
%idxRemove = inactivePlanes;
%for j = 2:dim+1
%idxRemove = [idxRemove inactivePlanes+n*(j-1)];
%end
%params(idxRemove) = []; influence(inactivePlanes) = [];

% output 
logLike = TermA*N; runTime = toc(totTime);

% save some statistics
statistics.iterations = iter; statistics.influence = influence;
statistics.timings.calcNewtonStep = timing.calcNewtonStep; statistics.timings.evalGrid = timing.evalGrid; statistics.timings.reduceHypers = timing.reduceHypers; statistics.timings.optimization = runTime;
statistics.numHypersHist = numHypersHist;

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
    options.lambdaSqEps = 10^-5;
end

if ~isfield(options,'intEps')
    options.intEps = 10^-3;
end

end
