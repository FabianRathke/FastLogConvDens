numSamples = 10000; dim = 2;
X = randn(numSamples,dim);
gamma = 1000;
optOptions = struct('verbose',2,'returnGrid',1);
t = tic;[aOpt bOpt logLike statistics] = lcdFast(X,gamma,optOptions); time = toc(t);

% grid and convex hull parameters for plotting (only for 2-D)
if dim == 2
	[N M grid weight gridSize] = setGridDensity(X,dim,0,struct('N',15,'M',15));
	[XXX YYY] = meshgrid(grid(1,:),grid(2,:));
	XXX = reshape(XXX,1,[]); YYY = reshape(YYY,1,[]);
	cvh = convhulln(X); cvh = cvh(:,1);

	evalInner = gamma*(aOpt*[XXX; YYY] + repmat(bOpt,1,length(XXX)));
	estDens = reshape(sum(exp(evalInner-repmat(max(evalInner),length(bOpt),1))).^(-1/gamma).*exp(-max(evalInner)/gamma),size(grid,2),[]);
	plotMTContour(X,grid(1,:),grid(2,:),estDens,[cvh(:,1); cvh(1,1)],[],[],'',''); set(gcf,'visible','on');
end
