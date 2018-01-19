function out = enet_filter(z, lambda, v, gamma)
gamma = gamma(1);
lambda = lambda(1);
z = z(:);
len = length(z);
if nargin < 3; v = ones(len, 1); end
if isempty(v);  v = ones(len, 1); end
if gamma == 1;
    out = lasso_filter(z, lambda, v);
else
    out = lasso_filter(z, lambda*gamma, v);
    lam = lambda(1);
    out = (v./(v+lam*(1-gamma))).*out;
end