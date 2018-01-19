function out = scad_filter(z, lambda, v, gamma)
z = z(:);
len = length(z);
if (nargin < 3)||(isempty(v)) ; v = ones(len, 1); end
v = v(:);
gam = gamma(1);
if gam <= 2; error('SCAD does not work when gamma <= 2 !'); end
z = z(:);
lam = lambda(1);

%% output
if len == 1;
    if abs(z) > v .* gam * lam
        out = z / v;
    elseif abs(z) <= lam * (v + 1);
        out = lasso_filter(z, lam) / v;
    else
        out = lasso_filter(z, gam * lam / (gam-1)) ./ (v - 1 / (gam-1));
    end
else
    out = zeros(len, 1);
    % case 1
    ind1 = abs(z) > v .* gam * lam;
    out(ind1) = z(ind1) ./ v(ind1);
    % case 2
    ind2 = (abs(z) <= v .* gam * lam) & (abs(z) > lam .* (v + 1));
    s2 = lasso_filter(z(ind2), gam * lam / (gam-1));
    out(ind2) = s2 ./ (v(ind2) - 1 / (gam-1));
    % case 3
    ind3 = ~(ind1 + ind2);
    s3 = lasso_filter(z(ind3), lam);
    out(ind3) = s3 ./ v(ind3);
end