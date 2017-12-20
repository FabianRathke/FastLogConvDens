nfunction [params logLike statistics] = newtonBFGSLInit(params,X,gamma,gridParams)

options = struct();

if ~isfield(options,'lambdaSqEps')
    options.lambdaSqEps = 10^-5;
end

if ~isfield(options,'intEps')
    options.intEps = 10^-3;
end


dim = size(X,2); lenP = length(params);
n = length(params)/(dim+1); N = length(X); M = length(gridParams.YIdx);
a = params(1:dim*n); a = reshape(a,[],dim); b = params(dim*n+1:end);
influence = zeros(n,1); statistics = struct();

% line search parameters
alpha = 10^-4; beta = 0.1;
[gradA gradB TermA TermB] = calcGradFloatAVX(single(X),single(gridParams.grid(1:dim)),single(a)',single(b),gamma,gridParams.weight,single(gridParams.delta),influence,single(sW),gridParams.gridSize,gridParams.YIdx); grad = double(gradA+gradB);

% the initial step is pure gradient descent
newtonStep = -grad;

% LBFGS variables
m = 20; % number of previous gradients that are used to calculate the inverse Hessian
s_k = zeros(lenP,m);
y_k = zeros(lenP,m);
sy = zeros(1,m); syInv = zeros(1,m);
updateList = 0; updateListInterval = 5;

for iter = 1:10000
	numHypersHist(iter) = length(b); inactivePlanes = [];

	lambdaSq = grad'*-newtonStep;
	if lambdaSq < 0
		newtonStep = -grad;
		lambdaSq = norm(grad)^2;
	end

	if lambdaSq > 10^5
		newtonStep = -grad;
		lambdaSq = norm(grad)^2;
	end

	step = 1;
	% objective function value before the step
	TermAOld = TermA; TermBOld = TermB; funcVal = TermAOld + TermBOld; gradOld = grad;
	% new parameters
	paramsNew = params + step*newtonStep; aNew = paramsNew(1:dim*n); aNew = reshape(aNew,[],dim); bNew = paramsNew(dim*n+1:end);
	% funcVal after the step
	[gradA gradB TermA TermB] = calcGradFloatAVX(single(X),single(gridParams.grid(1:dim)),single(aNew)',single(bNew),gamma,gridParams.weight,single(gridParams.delta),influence,single(sW),gridParams.gridSize,gridParams.YIdx); grad = double(gradA+gradB);
	funcValStep = TermA + TermB;

	% backtracking line search with Wolfe conditions (see Nocedal Chapter 3)
	while isnan(funcValStep) || isinf(funcValStep) || funcValStep > funcVal - step*alpha*lambdaSq 
		if step < 10^-9
			break
		end
		step = beta*step;
		paramsNew = params + step*newtonStep; aNew = paramsNew(1:dim*n); aNew = reshape(aNew,[],dim); bNew = paramsNew(dim*n+1:end);

		[gradA gradB TermA TermB] = calcGradFloatAVX(single(X),single(gridParams.grid(1:dim)),single(aNew)',single(bNew),gamma,gridParams.weight,single(gridParams.delta),influence,single(sW),gridParams.gridSize,gridParams.YIdx); grad = double(gradA+gradB); 
		funcValStep = TermA + TermB;
	end
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
end

% output 
logLike = TermA*N; 
end
