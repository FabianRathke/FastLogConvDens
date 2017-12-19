% makeRiemannGrid: Creates a grid with boundaries defined by box and n points along each dimension
% 	approach: 	'midpoint' 
%				'trapeziod'
%				'simpson'
%				'clenshaw'
%				'sparseDKP'
%
% [XGrid w] = makeGridND(box,n,approach)

function [XGrid w] = makeGridND(box,n,approach,basic)

if nargin < 4
	basic = 0;
end

if strcmp(approach,'simpson') && mod(n,2)
	error('For the Simpson approach an even n is required\n');
end

dim = size(box,1);

% sparse grids
if strcmp(approach,'sparseDKP') 
    [XGrid w] = nwspgr('KPU',dim,n);
	XGrid = XGrid'; w = w'*prod(box(:,2)-box(:,1));
	for k = 1:dim
		XGrid(k,:) = XGrid(k,:).*(box(k,2)-box(k,1)) + box(k,1);
	end
% dense grids
else
	% create 1-D grids and weights
	h = (box(:,2)-box(:,1))/n;
	for i = 1:dim
		% trapezoid, simpson
		if strcmp(approach,'simpson') || strcmp(approach,'trapezoid')
			grid(i,:) = linspace(box(i,1),box(i,2),n+1);
		% midpoint rule
		elseif strcmp(approach,'midpoint')
			interval = linspace(box(i,1),box(i,2),n+1);
			grid(i,:) = interval(1:end-1)+(interval(2)-interval(1))/2;
		elseif strcmp(approach,'clenshaw')
			[grid(i,:) wBasis(i,:)] = fclencurt(n,box(i,1),box(i,2));
		end

		% 1-D weights
		if strcmp(approach,'midpoint') 
			wBasis(i,:) = ones(1,n)*h(i);
		elseif strcmp(approach,'trapezoid')
			wBasis(i,:) = [1 ones(1,n-1)*2 1]*h(i)/2;
		elseif strcmp(approach,'simpson')
			if n < 4
				wBasis(i,:) = [1 4 1]*h(i)/3;
			else
				wBasis(i,:) = [1 repmat([4 2],1,(n-2)/2) 4 1]*h(i)/3;
			end
		end
	end
	if basic 
		XGrid = grid;
		w = wBasis;
	else
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

		w = wBasis(1,:);
		for i = 2:dim
			w = reshape(w'*wBasis(i,:),1,[]);
		end
	end
end
