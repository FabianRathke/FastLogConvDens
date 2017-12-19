function [XGrid] = makeGridND(box,n)

dim = size(box,1);

% create 1-D grids and weights
for i = 1:dim
	grid(i,:) = linspace(box(i,1),box(i,2),n+1);
end

evalString = '['; evalString2 = ''; YIdxString = '[';

for i = 1:dim
	if i < dim
		evalString = [evalString 'X' num2str(i) ' '];
		evalString2 = [evalString2 'grid(' num2str(i) ',:),'];
		YIdxString = [YIdxString 'Y' num2str(i) ', '];
	else
		evalString = [evalString 'X' num2str(i) ']'];
		evalString2 = [evalString2 'grid(' num2str(i) ',:)'];
		YIdxString = [YIdxString 'Y' num2str(i) ']'];
	end
end

eval([evalString ' = ndgrid(' evalString2 ');']);

XGrid = [];
for i = 1:dim
	eval(['X' num2str(i) ' = reshape(X' num2str(i) ',1,[]);']);
	eval(['XGrid = [XGrid; X' num2str(i) '];']);
	eval(['clear X' num2str(i)]);
end
