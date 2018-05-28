function plot2DDens(X, aOpt, bOpt)
	gamma = 1000
    [N M grid weight gridSize] = setGridDensity(X,2,0,struct('N',10,'M',10));
    [XXX YYY] = meshgrid(grid(1,:),grid(2,:));
    XXX = reshape(XXX,1,[]); YYY = reshape(YYY,1,[]);
    cvh = convhulln(X); cvh = cvh(:,1);
    [evalInner, idHypers] = min(-(aOpt*[XXX; YYY] + repmat(bOpt,1,length(XXX))));
	plotMTContour(X,grid(1,:),grid(2,:),reshape(exp(evalInner),size(grid,2),[]),[cvh(:,1); cvh(1,1)],[],[],getenv('JOURNALPLOTS'),'Density'); set(gcf,'visible','on');
end
