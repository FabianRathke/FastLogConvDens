function [densEst params logLike] = gaussianEM(X,posterior)

logLike = 0;
[N dim] = size(X);
N_k = sum(posterior);
K = length(N_k);

for iter = 1:1000
    % M-Step (update density parameters and mixing property tau)
    for j = 1:K
        mu(:,j) = sum(repmat(posterior(:,j),1,dim).*X)/N_k(j);
        Sigma(:,:,j) = (repmat(posterior(:,j),1,dim).*(X-repmat(mu(:,j)',N,1)))'*(X-repmat(mu(:,j)',N,1))/N_k(j);
        tau(j) = N_k(j)/N;
    end

    % E-Step
    % evaluate marginal densities p(x|z,\beta)
    for j = 1:K
        densEst(:,j) = mvnpdf(X,mu(:,j)',squeeze(Sigma(:,:,j)));
    end

    % update posterior probabilities p(z|x,\beta)
    for j = 1:K
        posterior(:,j) = tau(j)*densEst(:,j)./sum(repmat(tau,N,1).*densEst,2);
        N_k(j) = sum(posterior(:,j));
    end

    % evaluate the log likelihood p(X|\beta)
    logLike(iter+1) = sum(log(sum(repmat(tau,N,1).*densEst,2)));
    fprintf('%d: %.4f\n',iter,logLike(iter+1));

    if abs(logLike(iter+1)-logLike(iter))<10^-6;
        break
    end
end

params.mu = mu;
params.Sigma = Sigma;
params.tau = tau;
