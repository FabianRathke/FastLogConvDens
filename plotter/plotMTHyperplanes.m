function plotMTHyperplanes(X,gridX,gridY,XT,T,cmap,estimate,k,XLim,YLim,folder,filename,contourLines)

if ~exist('contourLines','var')
    contourLines = exp([-2.1:-0.5:-40]);
end

figure('visible','on');
map = eval(sprintf('colormap(%s(length(T)));',cmap));
[C h] = contour(gridX,gridY,estimate,contourLines,'Color','black'); hold on
hold on;
idx = randperm(length(T));
for i=1:length(T)
	patch(XT(T(i,:),1),XT(T(i,:),2),map(idx(i),:),'EdgeColor','white');
end

%set(gcf,'Colormap',paruly(length(k)-1));
%set(h,'AlphaData',0.5);
set(gca,'YDir','normal');
%[C h] = contour(gridX,gridY,estimate,contourLines,'Color','black'); %clabel(C,h);
cornerPoints = [gridX(1)-0.05 gridY(1)-0.05; gridX(end)+0.05 gridY(1)-0.05; gridX(end)+0.05 gridY(end)+0.05; gridX(1)-0.05 gridY(end)+0.05];
% find the convex hull points closest to each corner
for i = 1:4
	dist = sqrt(sum((X(k,:)-repmat(cornerPoints(i,:),length(k),1))*[gridY(end)-gridY(end-1) 0; 0 gridX(end)-gridX(end-1)].*(X(k,:)-repmat(cornerPoints(i,:),length(k),1)),2));
	[AA BB] = min(dist);
	minDist(i) = BB;
end
% create patches containing corners, closest points, and all intermediate points
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
% draw convex hull
plot3(X(k,1),X(k,2),ones(length(k),1)*2,'-g');
% plot data points
plot3(X(:,1),X(:,2),ones(size(X,1),1)*3,'ro','MarkerSize',2,'MarkerFaceColor','red');
% crop plot to convex hull
if length(XLim) > 0
%	set(gca,'XLim',XLim);
%	set(gca,'YLim',YLim);
else
	set(gca,'XLim',[min(X(:,1)) max(X(:,1))]);
	set(gca,'YLim',[min(X(:,2)) max(X(:,2))]);
end
set(gca,'XLim',get(gca,'XLim')+[-0.05 0.05]);
set(gca,'YLim',get(gca,'YLim')+[-0.05 0.05]);
set(gca,'box','off');

if ~strcmp(filename,'')
	set(get(gca,'parent'),'position',[200 50 600 450]);
	set(gcf, 'paperpositionmode', 'auto');
	print([folder filename '.eps'],'-depsc2');
end

