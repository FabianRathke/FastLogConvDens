function plotMTContour(X,classID,IDX,gridX,gridY,estimate,k,folder,filename,plotPatches,contourLines)

if ~exist('contourLines','var')
	contourLines = exp([-2.5:-0.5:-40]);
end

if ~exist('plotPatches','var')
	plotPatches = 1;
end

[val idx] = max(estimate); idx = squeeze(idx);
[A B] = find(idx(1:end-1,:)-idx(2:end,:));
BNew1 = 1:max(B);
BNew2 = 1:max(B);
ANew1 = zeros(size(BNew1));
ANew2 = zeros(size(BNew1));
figure('visible','on'); hold on; 
for i = 1:max(B)
	[idx] = find(B==i);
	if length(idx) == 0
		BNew1(BNew1==i) = [];
		BNew2(BNew2==i) = [];
	elseif length(idx) == 2
		ANew1(BNew1==i) = max(A(idx));
		ANew2(BNew2==i) = min(A(idx));
	else
		if abs(ANew1(BNew1==i-1)-A(idx)) < abs(ANew2(BNew2==i-1)-A(idx))
			ANew1(BNew1==i) = A(idx);
			BNew2(BNew2==i) = [];
		else
			ANew2(BNew2==i) = A(idx);
			BNew1(BNew1==i) = [];
		end
	end
end	
ANew1 = ANew1(1:length(BNew1))
ANew2 = ANew2(1:length(BNew2))
%plot(gridX(BNew2),(gridY(ANew2)+gridY(ANew2+1))/2,'-b','LineWidth',2)
%plot(gridX(BNew1),(gridY(ANew1)+gridY(ANew1+1))/2,'-b','LineWidth',2)

estimate = squeeze(sum(estimate));
[C h] = contour(gridX,gridY,estimate,contourLines); hold on; %clabel(C,h);
set(h,'LineWidth',1.5);
cornerPoints = [gridX(1) gridY(1); gridX(end) gridY(1); gridX(end) gridY(end); gridX(1) gridY(end)];
% find the convex hull points closest to each corner
for i = 1:4
	dist = sqrt(sum((X(k,:)-repmat(cornerPoints(i,:),length(k),1))*[gridY(end)-gridY(end-1) 0; 0 gridX(end)-gridX(end-1)].*(X(k,:)-repmat(cornerPoints(i,:),length(k),1)),2));
	[AA BB] = min(dist);
	minDist(i) = BB;
end
% create patches containing corners, closest points, and all intermediate points
if plotPatches
	order = [1:4 1];
	for i = 1:length(minDist)
		pointsX = cornerPoints(i,1); pointsY = cornerPoints(i,2);
		if minDist(order(i)) - minDist(order(i+1)) > 0
			points = [minDist(order(i)):length(k) 1:minDist(order(i+1))];
		else
			points = [minDist(order(i)):minDist(order(i+1))];
		end
		pointsX = [pointsX X(k(points),1)']; pointsY = [pointsY X(k(points),2)'];
		pointsX = [pointsX cornerPoints(order(i+1),1)]; pointsY = [pointsY cornerPoints(order(i+1),2)];
		h = patch(pointsX,pointsY,ones(1,length(pointsX)),'white');
		set(h,'EdgeColor','white');
	end
end
% draw convex hull
plot3(X(k,1),X(k,2),ones(length(k),1)*2,'-','Color','green');
% plot data points

markerType = 'so';
markerSize = [3 4];
for i = 0:1
%	plot(X(classID==i,1),X(classID==i,2),markerType(i+1),'MarkerFaceColor',[0.7 0.7 0.7],'MarkerEdgeColor',[0.7 0.7 0.7],'MarkerSize',2);
	idxA = (classID==i) & (IDX'==i+1);
	idxB = (classID==i) & ~(IDX'==i+1);
	if sum(idxA) > sum(idxB)
		idxCorrect = idxA; idxIncorrect = idxB;
	else
		idxCorrect = idxB; idxIncorrect = idxA;
	end
	% correctly labeled
	hPlot = plot(X(idxCorrect,1),X(idxCorrect,2),['g' markerType(i+1)],'MarkerSize',1,'MarkerFaceColor',[0.7 0.7 0.7],'MarkerEdgeColor',[0.7 0.7 0.7]);
	% incorrectly labeled
	hPlot = plot(X(idxIncorrect,1),X(idxIncorrect,2),['r' markerType(i+1)],'MarkerSize',markerSize(i+1));
	if i == 0
		set(hPlot,'MarkerFaceColor','red');
	end
end
box on;
XLim = [min(X(:,1))+min(X(:,1))*0.025 max(X(:,1))+max(X(:,1))*0.025];
YLim = [min(X(:,2))+min(X(:,2))*0.025 max(X(:,2))+max(X(:,2))*0.025];

% crop plot to convex hull
if length(XLim) > 0
	set(gca,'XLim',XLim);
	set(gca,'YLim',YLim);
end

if ~strcmp(filename,'')
	set(get(gca,'Parent'),'Position',[200 50 400 350]);
	set(gcf, 'PaperPositionMode', 'auto');
	print([folder filename '.eps'],'-depsc2');
end
