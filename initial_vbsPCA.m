function [outU,outS,outV] = initial_vbsPCA(X, r)
% beta is the quantile
beta = 0.95;
[n, p] = size(X);
% Subset Selection
delta = quantile(abs(X(:)),beta);
inx = (abs(X)<=delta);
Y = zeros(size(X));
Y(inx) = X(inx).^2;
Y(~inx) = 2*delta*X(~inx) - delta^2;
I = SubsetSelection(Y,0.05);
J = SubsetSelection(Y',0.05);
Xsub = X(I,J);
% Reduced SVD
[U,S,V] = svd(Xsub);
U = U(:,1:r);
outS = S(1:r,1:r);
V = V(:,1:r);
% Zero padding
outV = zeros(p,r);
outV(J,:) = V;
outU = zeros(n,r);
outU(I,:) = U;
% loading = pca(Xsub);
% loading = loading(:,1:r);
% thre = 1.4826*median(abs(bsxfun(@minus,loading,median(loading))));
% temp = bsxfun(@minus,abs(loading),thre);
% loading(temp<0) = 0;
end

%- SubsetSelection
function output = SubsetSelection(Y,alpha)
T = sum(Y,2);
mu = median(T);
sigma = 1.4826*median(abs(T-mu));
T = (T-mu)/sigma;
P = 1-cdf('Normal',T,0,1);
output = HolmBonferroni(P,alpha);
if isempty(output)
    output = 1:length(P);
end
end
%- SubsetSelection
%- HolmBonferroni
function I = HolmBonferroni(P,alpha)
if isempty(alpha); alpha = 0.05;end
m = length(P);
[ps, pi] = sort(P);
k = 1:m;
thre = alpha./(m+1-k');
pk = ps <= thre;
I = pi(pk);
end
%- HolmBonferroni
