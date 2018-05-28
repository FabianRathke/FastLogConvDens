function [X Y meanX stdX] = initDataMaxAffine(dim,numSamples,options)

if nargin < 3
	options = struct();
end
options = setDefaults(options,dim);

Y = [];
% generate sample
if options.student
%   loadStudentCVS;
    load ~/student.mat
else
    if (~isfield(options,'X'))
		if strcmp(options.distribution,'normal')
       	 	fprintf('Drew %d samples from a multivariate normal distribution with mean zero\n',numSamples);
		    X = mvnrnd(zeros(1,dim),options.covMat,numSamples);
			Y = mvnpdf(X,zeros(1,dim),options.covMat);
        elseif strcmp(options.distribution,'gamma')
			fprintf('Drew %d samples from a gamma distribution\n',numSamples);
            X = gamrnd(options.shape,options.scale,numSamples,1);
			Y = gampdf(X,options.shape,options.scale);
        elseif strcmp(options.distribution,'beta')
			fprintf('Drew %d samples from a beta distribution\n',numSamples);
            X = betarnd(options.alpha,options.beta,numSamples,1);
			Y = gampdf(X,options.alpha,options.beta);
        end
    else
        fprintf('Used data from options.X\n');
        X = options.X;
    end
end
% normalize data
meanX = mean(X);
stdX = std(X);
if options.normalize
%	X = (X - repmat(mean(X),length(X),1))./repmat(std(X),length(X),1);
	X = X-repmat(mean(X),length(X),1);
else
	meanX = 0;
	stdX = 1;
end

if ~options.keepBox
    for i = 1:dim
        box(i,:) = [min(X(:,i)) max(X(:,i))];
    end
else
    box = options.box;
end

% save results for cule
filename = options.filename;
if options.saveToMat
	save(sprintf('~/%s.mat',options.saveName),'X','filename');
end

end



function options = setDefaults(options,dim) 

if (~isfield(options,'filename'))
	options.filename = 'yreturnND';
end

if (~isfield(options,'saveName'))
	options.saveName = 'yND';
end

if (~isfield(options,'normalize'))
	options.normalize = 0;
end

if (~isfield(options,'distribution'))
	options.distribution = 'normal';
end

if (~isfield(options,'student'))
	options.student = false;
end

if (~isfield(options,'keepBox'))
	options.keepBox = false;
end

if (~isfield(options,'covMat'))
	% sample positive definite matrix
	df = 10;
	options.covMat = wishrnd(eye(dim),df)/df;
%	options.covMat = eye(dim);
end

if (~isfield(options,'methodGrid'))
	options.methodGrid = 'riemann';
end

end
