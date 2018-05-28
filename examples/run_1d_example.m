cule = 1 % runs also code for the R package `LogConcDEAD` by Cule et al.
duembgen = 1 % runs also code for the package `logcondens` by Duembgen et al

% draw sample (automatically saves the result to a matfile for usage in R) 
numSamples = 500; dim = 1;
X = initData(dim,numSamples,struct('distribution','normal', 'saveToMat', true));

% optimization parameters (as defined in the paper)
gamma = 1000;
optOptions = struct('verbose',2,'cutoff',10^-1,'method',@newtonBFGSL,'lambdaSqEps',10^-7,'intEps',10^-3);

% run optimization
t = tic;[aOpt bOpt logLike gridParams] = lcdFast(X,gamma,optOptions); time(1) = toc(t);
fprintf('Parametrization our approach: %d hyperplanes\n', length(bOpt))

% plot resulting estimate
plotSparseParams(X, gridParams) 

% get estimate for Cule et al
if cule
	% runs a R script (Install R and packages 'LogConcDEAD' and 'R.matlab'
	system(['Rscript /path/to/lib/scripts/runCule.R yND']);
	load ~/yreturnND.mat

	colormap = lines;
	fontsize = 16;

	x = X(unique(triangulation(:)));
	x = sort(x);
	y = min(b*x' - repmat(beta,1,length(x)));

	figure; hold on; clear hPlot;
	for i = 1:length(x)-1
		hLine = line([x(i) x(i+1)],[y(i) y(i+1)]);
		set(hLine,'Color',colormap(i,:),'LineWidth',2);
	end
	for i = 1:length(x)-1
		plot(x(i),y(i),'.r','MarkerSize',7);
	end
	plot(x(end),y(end),'.r','MarkerSize',7);
	hTitle = title(sprintf('Parametrization Cule et al.'));
	fprintf('Parametrization Cule et al.: %d hyperplanes\n', length(beta))
	makePlotsNicer;
	set(gca,'YLim',[floor(min(y)) round(max(y)+1)],'XLim',[min(X) max(X)])
end

% get estimates for Duembgen et al
if duembgen
	system(['Rscript /path/to/lib/scripts/runRufibach.R yND']);
	load ~/yreturnND.mat
	fprintf('Parametrization Duembgen et al.: %d hyperplanes\n', length(idxKnots)-1)

	idxKnots = find(knots);
	figure; hold on;
	for i=1:length(idxKnots)-1
		hLine = line([xRufi(idxKnots(i)) xRufi(idxKnots(i+1))],[phi(idxKnots(i)) phi(idxKnots(i+1))]);
		set(hLine,'Color',colormap(i,:),'LineWidth',2);
	end
	for i =1:length(idxKnots)-1
		plot(xRufi(idxKnots(i)),phi(idxKnots(i)),'.r','MarkerSize',7);
	end
	plot(xRufi(idxKnots(end)),phi(idxKnots(end)),'.r','MarkerSize',7);
	makePlotsNicer
	set(gca,'YLim',[floor(min(y)) round(max(y)+1)],'XLim',[min(X) max(X)])
end
