clear all
K = 2;
classes = 2;
%wisconsin
usps_data
%iris

save ~/yEM.mat DP dataName classes
% *** Init using hierarchical clustering as in Cule et al. ***
system(['Rscript ' getenv('WORKHOME') '/MyCode/LogConcave/code/scripts/culeInitEM.R']);
load ~/yEMInit.mat
N_k = sum(posterior); tau = N_k/length(DP);
logLikeInit = sum(log(sum(repmat(tau,length(DP),1).*exp(y),2)));

% create a grid for plotting
[X] = initDataMaxAffine(size(DP,2),150,struct('X',DP,'normalize',0));

grid = [linspace(min(X(:,1)),max(X(:,1)),100); linspace(min(X(:,2)),max(X(:,2)),100)];
[XXX YYY] = meshgrid(grid(1,:),grid(2,:)); XXX = reshape(XXX,1,[]); YYY = reshape(YYY,1,[]);
cvh = convhulln(DP(:,[1 2]));

% plot the data points
%color = [216/256 63/256 190/256; 154/256 216/256 63/256; 35/256 187/256 255/256];
%markerType = {'s','o','d'};
%figure; hold on;
%for i = 0:max(classID)
%	if i == 0
%		plot(X(classID==i,1),X(classID==i,2),markerType{i+1},'MarkerFaceColor',color(i+1,:),'MarkerEdgeColor',color(i+1,:),'MarkerSize',4); hold on;
%	else
%		plot(X(classID==i,1),X(classID==i,2),markerType{i+1},'MarkerEdgeColor',color(i+1,:),'MarkerSize',4); hold on;
%	end
%end
%
%legend(classNames);
%XLim = [min(X(:,1))+min(X(:,1))*0.025 max(X(:,1))+max(X(:,1))*0.025]; set(gca,'XLim',XLim);
%YLim = [min(X(:,2))+min(X(:,2))*0.025 max(X(:,2))+max(X(:,2))*0.025]; set(gca,'YLim',YLim);
%
%set(get(gca,'Parent'),'Position',[200 50 400 400]);
%set(gcf, 'PaperPositionMode', 'auto');
%print([getenv('JOURNALPLOTS') dataName '_' 'data.eps'],'-depsc2');


% **** GAUSSIAN EM ********
[densEstGauss paramsGauss logLike] = gaussianEM(DP,posterior);
% missclassifaction rate
[~,IDXGauss] = max((repmat(paramsGauss.tau,N,1).*densEstGauss)');
fprintf('# Missclassified: %d\n',min(N-sum((IDXGauss'==1)==(classID==0)),N-sum((IDXGauss'==2)==(classID==0))));
% eval for plots
evalGridGauss = evalDensGauss([XXX; YYY],paramsGauss);
plotEMContour(X(:,[1 2]),classID,IDXGauss,grid(1,:),grid(2,:),evalGridGauss,[cvh(:,1); cvh(1)],getenv('JOURNALPLOTS'),['/' dataName '_' num2str(K) '_gaussEM'],0); set(gcf,'visible','on')

% *** LOG-CONCAVE EM ***
t = tic; [densEstLogConcave paramsLogConcave logLike] = lcdFastEM(DP,posterior,struct('verbose',1,'classID',classID,'fast',0)); time = toc(t);
save(sprintf('~/yEMResult-Ours-%s-%dD.mat',dataName,K),'densEstGauss','paramsGauss','densEstLogConcave','paramsLogConcave','time');
% missclassifaction rate
[~,IDXLog] = max((repmat(paramsLogConcave.tau,N,1).*densEstLogConcave)');
fprintf('# Missclassified: %d\n',min(N-sum((IDXLog'==1)==(classID==0)),N-sum((IDXLog'==2)==(classID==0))));
[evalDens evalGrid] = evalDensLogConcave(X,paramsLogConcave,[1 2]);
plotEMContour(X(:,1:2),classID,IDXLog,evalGrid(1,:),evalGrid(2,:),evalDens,[cvh(:,1); cvh(1)],getenv('JOURNALPLOTS'),['/' dataName '_' num2str(K) '_logConcaveEM']); set(gcf,'visible','on')

% Cule Result
system('Rscript ~/Documents/Code/MyCode/MassTransport/maxAffine/culeEM.R');
load(sprintf('~/yEMResult-%s-%dD.mat',dataName,K))
% missclassifaction rate
[~,IDXCule] = max((repmat(props',length(logf),1).*exp(logf))');
fprintf('# Missclassified: %d\n',min(N-sum((IDXCule'==1)==(classID==0)),N-sum((IDXCule'==2)==(classID==0))));
fprintf('Log-Likelihood: %.4f\n',sum(log(sum(repmat(props',length(logf),1).*exp(logf),2))))

evalGridCule = evalDensCule(X,[XXX;YYY],logf,props);
plotEMContour(X,classID,IDXCule,grid(1,:),grid(2,:),reshape(max(evalGridCule),size(grid,2),[]),[cvh(:,1); cvh(1)],getenv('JOURNALPLOTS'),[dataName '_' num2str(K) '_logCuleEM']); set(gcf,'visible','on')
