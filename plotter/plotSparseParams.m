function plotSparseParams(X, gridParams)

aOpt = gridParams.aSparse;
bOpt = gridParams.bSparse;
ACVH = gridParams.ACVH;
bCVH = gridParams.bCVH;
cvh = gridParams.cvh;
idx = randperm(length(aOpt));
aOpt = aOpt(idx,:); bOpt = bOpt(idx);

if size(X,2) == 1
	colormap = lines;
	[aOpt idxSort] = sort(aOpt);
	bOpt = bOpt(idxSort);

	% plot hyperplanes
	x = [max(X); (bOpt(2:end)-bOpt(1:end-1))./(aOpt(1:end-1)-aOpt(2:end)); min(X)];
	x = sort(x);
	x(x > max(X)) = [];
	x(x < min(X)) = [];
	y = -max(aOpt*x' + repmat(bOpt,1,length(x)));

	evalAll =-(aOpt*[min(x) max(x)] + repmat(bOpt,1,2));
	figure; hold on;

	for i=1:length(aOpt)
		h = line([min(x) max(x)],evalAll(i,:));
		set(h,'LineStyle','-.','Color',[0.6 0.6 0.6],'LineWidth',0.5);
	end
	for i = 1:length(x)-1
		hLine = line([x(i) x(i+1)],[y(i) y(i+1)]);
		set(hLine,'Color',colormap(i,:),'LineWidth',2);
	end
	plot(x(end),y(end),'.r','MarkerSize',7);
	for i=1:length(x)-1
		plot(x(i),y(i),'.r','MarkerSize',7);
	end
	Interpreter = 'latex';
	hTitle = title('Sparse parametrization for $$-\hat{\varphi}_n(x) = \log(\hat{f}_n(x))$$', 'Interpreter','latex');
	set(gca,'YLim',[floor(min(y)) round(max(y)+1)],'XLim',[min(X) max(X)])
	set(get(gca,'Parent'),'Position',[200 50 500 400]);
	makePlotsNicer
elseif size(X,2) == 2
	% Find intersections of sparse hyperplanes by brute force (check all combinations)
	combsPure = nchoosek(1:length(aOpt),2);
	% calculate all intersections
	aI = aOpt(combsPure(:,1),:)-aOpt(combsPure(:,2),:);
	bI = bOpt(combsPure(:,1))-bOpt(combsPure(:,2));

	combs = nchoosek(1:length(aI),2);
	detA = 1./(aI(combs(:,2),2).*aI(combs(:,1),1) - aI(combs(:,2),1).*aI(combs(:,1),2));
	junctions = -repmat(detA,1,2).*[aI(combs(:,2),2).*bI(combs(:,1)) - aI(combs(:,1),2).*bI(combs(:,2)) -aI(combs(:,2),1).*bI(combs(:,1)) + aI(combs(:,1),1).*bI(combs(:,2))];
	junctions = sortrows(round(junctions*10^8)/10^8);
	[junctions IA IC] = unique(junctions,'rows');
	% find junctions that are part of three hyperplanes
	[YA,YB]=hist(IC,unique(IC));
	junctions = junctions(YA==3,:);

	% only points inside the convex hull
	idxInside = find(sum(ACVH*junctions' - repmat(bCVH,1,length(junctions))<=0)==length(bCVH));
	combs = combs(idxInside,:); junctions = junctions(idxInside,:);

	% find junctions between intersections of hyperplanes and the convex hull
	[XI YI] = (meshgrid(1:length(bI),1:length(bCVH)));
	combsCVH = [reshape(XI,[],1) reshape(YI,[],1)];
	detA = 1./(ACVH(combsCVH(:,2),2).*aI(combsCVH(:,1),1) - ACVH(combsCVH(:,2),1).*aI(combsCVH(:,1),2));
	junctionsCVH = -repmat(detA,1,2).*[ACVH(combsCVH(:,2),2).*bI(combsCVH(:,1)) + aI(combsCVH(:,1),2).*bCVH(combsCVH(:,2)) -ACVH(combsCVH(:,2),1).*bI(combsCVH(:,1)) - aI(combsCVH(:,1),1).*bCVH(combsCVH(:,2))];

	% only points on the convex hull
	idxInside = find(sum(ACVH*junctionsCVH' - repmat(bCVH,1,length(junctionsCVH))<=10^-8)==length(bCVH));
	combsCVH = combsCVH(idxInside,:); junctionsCVH = junctionsCVH(idxInside,:);

	% check which intersections belong to g(x) --> check if at least three hyperplanes are maximal at this point
	junctionsEval = aOpt*junctions'+repmat(bOpt,1,length(junctions));
	junctionsEval = round(junctionsEval*10^5)/10^5;
	idxMaxInside = sum(junctionsEval==repmat(max(junctionsEval),length(bOpt),1))==3;
	[val idx] = sort(junctionsEval(:,idxMaxInside));
	idx = idx(end-2:end,:);

	junctionsEval = aOpt*junctionsCVH'+repmat(bOpt,1,length(junctionsCVH));
	junctionsEval = round(junctionsEval*10^5)/10^5;
	idxMaxCVH = sum(junctionsEval==repmat(max(junctionsEval),length(bOpt),1))==2;
	[val idxCVH] = sort(junctionsEval(:,idxMaxCVH));
	idxCVH = idxCVH(end-1:end,:); 
	% calculate triangulation
	junctions = junctions(idxMaxInside,:); junctionsCVH = junctionsCVH(idxMaxCVH,:);

	[N M grid weight gridSize] = setGridDensity(X,2,0,struct('N',20,'M',20));
	[ACVH bCVH cvh] = calcCvxHullHyperplanes(X);

	[XXX YYY] = meshgrid(grid(1,:),grid(2,:));
	XXX = reshape(XXX,1,[]); YYY = reshape(YYY,1,[]);

	%evalInner = gamma*(aOpt*[XXX; YYY] + repmat(bOpt,1,length(XXX)));
	%estDens = reshape(sum(exp(evalInner-repmat(max(evalInner),length(bOpt),1))).^(-1/gamma).*exp(-max(evalInner)/gamma),size(grid,2),[]);
	%[val idHypers] = max(evalInner); idHypers = reshape(idHypers,size(grid,2),[]);
	%estDens = exp(-reshape(max(aOpt*[XXX; YYY] + repmat(bOpt,1,length(XXX))),size(grid,2),[]));

	[evalInner, idHypers] = min(-(aOpt*[XXX; YYY] + repmat(bOpt,1,length(XXX))));
	plotMTHyperplanesOld(X,grid(1,:),grid(2,:), reshape(idHypers, size(grid,2),[]), reshape(exp(evalInner),size(grid,2),[]),[cvh(:,1); cvh(1,1)],[],[],getenv('JOURNALPLOTS'),''); set(gcf,'visible','on'); hold on;
	Interpreter = 'latex';
	hTitle = title('Sparse parametrization for $$-\hat{\varphi}_n(x) = \log(\hat{f}_n(x))$$', 'Interpreter','latex');
	for i = 1:length(idx)
		for j = i+1:length(idx)
			if sum(ismember(idx(:,i),idx(:,j))) == 2
				hLine = line([junctions(i,1) junctions(j,1)],[junctions(i,2) junctions(j,2)]);
				set(hLine,'color',ones(1,3)*1,'LineWidth',1.5);
			end
		end
	end

	for i = 1:length(idx)
		for j = 1:length(idxCVH)
			if sum(ismember(idx(:,i),idxCVH(:,j))) == 2
				hLine = line([junctions(i,1) junctionsCVH(j,1)],[junctions(i,2) junctionsCVH(j,2)]);
				set(hLine,'color',ones(1,3)*1,'LineWidth',1.5);
			end
		end
	end

	% normalize Integral
%	X_ = [junctions; junctionsCVH];
%	yT = min(-aOpt*X_'-repmat(bOpt,1,length(X_)));
%	T = int32(convhulln([X_ yT'; X_ repmat(min(yT)-1,size(X_,1),1)]));
%	T(max(T,[],2)>length(yT),:) = [];
%
%	[aOptNew bOptNew integral changeB Ad Gd] = calcExactIntegral(X_',yT,T'-1,2,1);
%
%	aOptNew = aOptNew';
%	if norm(max(aOptNew*X_(1:min(size(X_,1),10),:)' + repmat(bOptNew,1,min(length(yT),10)),[],1) + yT(1:min(10,length(yT)))) > 10^-6
%		warning('Potential numerical problems when calculating the final set of hyperplanes --> Recommended to run the optimization again');
%	else
%		aOpt = aOptNew; bOpt = bOptNew;
%	end
else
	fprintf('Plotting is only supported for 1-D and 2-D\n');
end
