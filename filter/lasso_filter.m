function out = lasso_filter(z, lambda, v, gamma)
% gamma = [];
z = z(:);
len = length(z);
if nargin < 3; v = ones(len, 1); end
lam = lambda(1);

out = (abs(z) >= lam) .* (z - sign(z) .* lam);
out = out ./ v;
end