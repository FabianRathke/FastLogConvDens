# FastLogConvDens
Matlab Code for the R package fmlogcondens (https://cran.r-project.org/web/packages/fmlogcondens/). 
Estimates a log-concave density or a mixture thereof. Allows the fast processing of large data samples of 10000 points and more.


### Example
Estimate the log-concave density for a sample of 2500 points in 2-D and plot the resulting density:

```matlab
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

% Plot sparse parametrization (may take several minutes for >> 50 hyperplanes)
plotSparseParams(X, gridParams)
```

For more examples see the scripts in the `examples` directory.
