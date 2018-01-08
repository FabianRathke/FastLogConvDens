ACVH = gridParams.ACVH;
bCVH = gridParams.bCVH;
X = single(X);
n_ = size(X,1);
sW = single(ones(n_,1)/n_);
[N dim] = size(X);
%n = 2*dim; lenP = n*(dim+1);
%a = rand(n,dim)*0.1; b = rand(n,1);
%params = single([reshape(a,[],1); b]);
params = single(params);
box = double([min(X)' max(X)']);
save /home/fabian/Documents/Arbeit/Code/MyCode/LogConcave/code/newtonInit.mat X sW ACVH bCVH box params
X = double(X);
params = double(params);
