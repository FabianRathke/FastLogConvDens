% Demonstrate the power of our approach by running some heavy data examples. For comparison: the average runtime for Cule et al. for 10000 samples in 
% 2-D is about 4.5 hours and 10.5 hours for 3-D on a standard 4 core machine.

N = 10000;

for dim = 2:4
	X = randn(N, dim);

	% Parameters
	gamma = 1000;
	optOptions = struct('verbose',0,'cutoff',10^-1,'method',@newtonBFGSL,'lambdaSqEps',10^-7,'intEps',10^-3);

	% Perform the optimization
	t = tic;[aOpt bOpt logLike gridParams] = lcdFast(X,gamma,optOptions); time(1) = toc(t);
	fprintf('%d-D: Optimization finished in %.2f s\n', dim, time(1));
end
