function out = mcp_filter(z, lambda, v, gamma)
z = z(:);
len = length(z);
if (nargin < 3)||(isempty(v)) ; v = ones(len, 1); end
v = v(:);
gam = gamma(1);
if gam <= 1; error('MCP does not work when gamma <= 1 !'); end
z = z(:);
lam = lambda(1);

%% output
if len == 1
    if abs(z) > v .* gam * lam
        out = z / v;
    else
        out = lasso_filter(z, lam) / (v - 1 / gam);
    end
else
    out = zeros(len, 1);
    % case 1
    ind = abs(z) > v .* gam * lam;
    out(ind) = z(ind) ./ v(ind);
    % case 2
    s = lasso_filter(z(~ind), lam);
    out(~ind) = s ./ (v(~ind) - 1 / gam);
end