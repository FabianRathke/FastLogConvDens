function colors = loadColors(colorscheme)

color_set.light_yellow = [255 249 98];
color_set.yellow = [236 209 24];
color_set.orange = [249 197 15];
color_set.light_blue = [127 161 195];
color_set.blue = [20 88 157];
color_set.red = [205 45 11];
color_set.light_green = [156 233 60];
color_set.green = [52 174 15];
color_set.gray = [150 150 150];

if ~isempty(colorscheme)
    for i_ = 1:length(colorscheme)
        eval(sprintf('colors(%d,:) = color_set.%s;',i_,colorscheme{i_}));
    end
else
	% [default] {'yellow','orange','blue','light_blue'}
    colors = [236 209 24; 249 197 15; 20 88 157; 127 161 195; 205 45 11; 210 251 151; 255 249 98];
end

