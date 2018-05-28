% Draw samples
X = randn(2500, 2);

% Parameters
gamma = 1000;
optOptions = struct('verbose',2,'cutoff',10^-1,'method',@newtonBFGSL,'lambdaSqEps',10^-7,'intEps',10^-3);

% Perform the optimization
t = tic;[aOpt bOpt logLike gridParams] = lcdFast(X,gamma,optOptions); time(1) = toc(t);
fprintf('Optimization finished in %.2f s\n', time(1));

% Plot the resulting density
plot2DDens(X, aOpt, bOpt)

% Plot sparse parametrization (could be slow for >> 50 hyperplanes)
plotSparseParams(X, gridParams)
