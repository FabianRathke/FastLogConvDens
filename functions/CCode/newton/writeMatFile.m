ACVH = gridParams.ACVH;
bCVH = gridParams.bCVH;
box = double([min(X)' max(X)']);

params = reshape(params,[],size(X,2)+1);
randIdx = randperm(size(params,1));
params = params(randIdx,:);
params = reshape(params,[],1);
save /home/fabian/Documents/Arbeit/Code/MyCode/LogConcave/code/newtonInit.mat X sW ACVH bCVH box params
