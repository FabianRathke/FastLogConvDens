function [params gridParams statistics] = paramFitGammaOne(X,sampleWeights,ACVH,bCVH,cvh,optOptions) 
[n dim] = size(X);


[N M gridParams.grid gridParams.weight gridParams.gridSize] = setGridDensity(X,dim,1,struct());
gridParams.N = N; gridParams.M = M;
gridParams.delta = [gridParams.grid(:,2)-gridParams.grid(:,1)];
gridParams.ACVH = ACVH; gridParams.bCVH = bCVH;
gridParams.sparseGrid = makeGridND([min(X)' max(X)'],N,'trapezoid');
gridParams.sparseDelta = (gridParams.sparseGrid(:,end)-gridParams.sparseGrid(:,1))/gridParams.N;
[gridParams.YIdx gridParams.gridToBox gridParams.XToBox, gridParams.numPointsPerBox, gridParams.boxEvalPoints, gridParams.YIdxSub gridParams.subGrid gridParams.subGridIdx gridParams.boxIDs gridParams.XToBoxOuter] = makeGrid(gridParams.sparseGrid,[min(X) max(X)],ACVH,bCVH,N,M,dim,X,gridParams.sparseDelta);

[N dim] = size(X);
n = 10*dim; lenP = n*(dim+1);
a = rand(n,dim)*0.1; b = rand(n,1);
params = [reshape(a,[],1); b];

%yT = -log(kernelDens(X,X,sampleWeights));
%
%ft = exp(a*X' + repmat(b,1,N));
%h_ft = sum(ft); h_ft_inv = 1./h_ft;
%ft_x = zeros((dim+1)*n,N);
%g_ft = log(h_ft);
%
%for j = 1:dim
%    ft_x((1:n)+(j-1)*n,:) = ft.*repmat(X(:,j)'.*h_ft_inv.*(g_ft-yT'),n,1);
%end
%
%ft_x(end-n+1:end,:) = ft.*repmat(h_ft_inv.*(g_ft-yT'),n,1);
%grad = sum(ft_x,2);
%B = eye(lenP);
%alpha = 10^-4; beta = 0.1;
%
%for iter = 1:1000
%    newtonStep = B\-grad;
%    lambdaSq = grad'*-newtonStep;
%
%    % objective function value before the step
%    funcVal = sum((g_ft-yT').^2);
%
%    paramsNew = params + newtonStep; aNew = paramsNew(1:dim*n); aNew = reshape(aNew,[],dim); bNew = paramsNew(dim*n+1:end);
%    % objective function value after the step
%    ft = exp(aNew*X' + repmat(bNew,1,N));
%	g_ft = log(sum(ft));
%
%	funcValStep = sum((g_ft-yT').^2);
%    % backtracking line search
%    step = 1;
% 	while (funcValStep > funcVal - step*alpha*lambdaSq)
%        fprintf('\r%e',step);
%        if step < 10^-10
%            break
%        end
%        step = beta*step;
%        paramsNew = params + step*newtonStep; aNew = paramsNew(1:dim*n); aNew = reshape(aNew,[],dim); bNew = paramsNew(dim*n+1:end);
%
%        ft = exp(aNew*X' + repmat(bNew,1,N)); g_ft = log(sum(ft));    
%
%		funcValStep = sum((g_ft-yT').^2);
%    end
%	if abs(lambdaSq) < 10^-5
%        break
%    end
%
%    params = paramsNew; a = params(1:dim*n); a = reshape(a,[],dim); b = params(dim*n+1:end);
%	grad_old = grad;
%
%	h_ft = sum(ft); h_ft_inv = 1./h_ft;
%	g_ft = log(h_ft);
%
%	for j = 1:dim
%    	ft_x((1:n)+(j-1)*n,:) = ft.*repmat(X(:,j)'.*h_ft_inv.*(g_ft-yT'),n,1);
%	end
%
%	ft_x(end-n+1:end,:) = ft.*repmat(h_ft_inv.*(g_ft-yT'),n,1);
%	grad = sum(ft_x,2);
%
%    if (sum(isnan(grad)))
%        error('NaN in Grad');
%3    end
%
%    % modified BFGS for non-convex optimization (Li paper)
%    s = step*newtonStep;
%    gammaBFGS = grad - grad_old;
%    t = 1 +  max([-gammaBFGS'*s/norm(s).^2 0]);
%    y = gammaBFGS + t*norm(grad)*s;
%    Bs = B*s; sy = s'*y;
%    B = B + y*y'/(sy) - Bs*Bs'/(s'*Bs); 
%end


[optParams logLike statistics] = newtonBFGSLCInit(params,X,1,gridParams,struct('verbose',0,'reduceHyperplanes',0));
aOpt = reshape(optParams(1:10*dim*dim),[],dim); bOpt = optParams(10*dim*dim+1:end);

yT = -log(sum(exp(aOpt*X' + repmat(bOpt,1,length(X)))))';

% DEBUG: plot density function
%figure; 
%plot(X,exp(yT),'.');

idxCVH = unique(cvh);
T = int32(convhulln([X yT; X(idxCVH,:) repmat(min(yT(idxCVH))-1,length(idxCVH),1)]));
%T = int32(convhulln([X yT; X repmat(min(yT)-1,length(yT),1)]));
T(max(T,[],2)>length(yT),:) = [];
%T = int32(delaunayn(X));

numHypers = length(T);
[aOpt bOpt] = calcExactIntegral(X',yT,T'-1,dim,1);
aOpt = aOpt';
	
if (size(X,2) == 1) && length(bOpt) > 1000
	idxSelect = randperm(length(bOpt),1000);
	aOpt = aOpt(idxSelect);
	bOpt = bOpt(idxSelect);
end

params = [aOpt bOpt];
 
if mean(var(aOpt(randperm(length(bOpt),min(100,length(bOpt))),:))) < 10^-4 || length(params) == dim+1
	fprintf('#### Bad initialization due to small sample size, switch to kernel kensity based initialization ####\n');
	params = paramFitKernelDensity(X,optOptions.sampleWeights,gridParams,cvh);
end

