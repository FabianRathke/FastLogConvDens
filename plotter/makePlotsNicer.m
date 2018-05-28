legend boxoff

if ~exist('fontsize')
	fontsize = 12;
end

if ~exist('Interpreter')
	Interpreter = 'tex';
end

%set(gcf,'PaperPosition',[0.25 2.5 6.0 6.0/8*5.5]);
position = get(gca,'Position');
%set(gca,'Position',position + [0 .02 0 0 ]);
set( gca                       , ...
    'FontName'   , 'Helvetica' );
%set([hLegend, hTitle], ...
%    'FontName'   , 'AvantGarde');

set([gca]             , ...
    'FontSize'   , fontsize           );
if exist('hLegend')
	set([hLegend]             , ...
    	'FontSize'   , fontsize+1     , ...
		'Visible', 'on', ...
		'Interpreter', Interpreter, ...
		'Box', 'off');
%		'EdgeColor',[1 1 0.99]);

	% increase distance of text to labels
	children = get(hLegend,'Children');
	for i_ = 1:length(children)
		if (strcmp('text',get(children(i_),'Type')))
			position = get(children(i_),'Position');
			position(1) = position(1)+0.05;
			set(children(i_),'Position',position);
		end
	end
end

if exist('hXLabel')
	if (strcmp(Interpreter,'Latex'))
		fontsizeLocal = fontsize+3;
	else
		fontsizeLocal = fontsize;
	end
   	set(hXLabel,'FontSize',fontsizeLocal,'Interpreter',Interpreter);
	position = get(get(gca,'XLabel'),'Position');
%	set(get(gca,'XLabel'),'Position',position - [0 0.02 0]);
end

if exist('hYLabel')
	if (strcmp(Interpreter,'Latex'))
		fontsizeLocal = fontsize+3;
	else
		fontsizeLocal = fontsize;
	end
	set(hYLabel,'FontSize',fontsizeLocal,'Interpreter',Interpreter); 
%	position = get(get(gca,'YLabel'),'Position');
%	set(get(gca,'YLabel'),'Position',position - [0.1 0 0]);
end

if exist('hTitle')
	set(hTitle, 'FontSize',fontsize+2,'FontWeight','normal','Color',[.3 .3 .3],'Interpreter',Interpreter);

	%position = get(hTitle,'Position');
	%set(hTitle,'Position',[position(1) position(2)+position(2)*0.005 position(3)]);	
end


set(gca, ...
  'Box'         , 'off'     , ...
  'TickDir'     , 'out'     , ...
  'XColor'      , [.3 .3 .3], ...
  'YColor'      , [.3 .3 .3], ...
  'YGrid'		,  'on');


if ~exist('colors')
	if exist('colorscheme')
		colors = loadColors(colorscheme);
		%for i_ = 1:length(colorscheme)
		%	eval(sprintf('colors(%d,:) = color_set.%s;',i_,colorscheme{i_}));
		%end
	else
		%[default]: {'yellow','light_blue',
		colors = [236 209 24; 127 161 195; 245 82 24; 20 88 157; 156 233 60;  249 197 15; 210 251 151; 255 249 98; 0 0 0; 0 0 0];
		%colors = [200 200 200; 40 40 40];
	end
end

if (exist('hBar'))
	for i_ = 1:length(hBar)
		set(hBar(i_),'FaceColor',getColor(colors(i_,:)));
		set(hBar(i_),'EdgeColor',[0.5 0.5 0.5]);
	end
end

if (~exist('markerScheme'))
	markerScheme = {'v','s','o','d','^'};
end
if (~exist('setMarkers'))
	setMarkers = 0;
end

if (exist('hPlot'))
	set(hPlot(:),'LineWidth',1.5);

	for i_ = 1:length(hPlot)
		set(hPlot(i_),'MarkerFaceColor',getColor(colors(i_,:)));
		set(hPlot(i_),'MarkerEdgeColor',getColor(colors(i_,:)));
		set(hPlot(i_),'Color',getColor(colors(i_,:)));
		if setMarkers
			set(hPlot(i_),'Marker',markerScheme{i_});
		end
	end
end

%figsToDelete = {'hPlot','hBar','hTitle','hLegend','hXLabel','hYLabel'}
%for counter = 1:length(figsToDelete)
%	eval(sprintf('clear %s',figsToDelete{counter}));
%end
