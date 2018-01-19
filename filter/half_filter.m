function out = half_filter(z, lambda, v, gamma)
% gamma = [];
z = z(:);
len = length(z);
if nargin < 3; v = ones(len, 1);  end
v = v(:);
lam = lambda(1);

out = zeros(len, 1);
ind = abs(z) > (0.75 * lam^(2/3) .* v.^(1/3));
out_temp = cos((2 / 3) * (pi - half_phi(z(ind),lam,v(ind))));
out(ind) = (2/3) * (z(ind)./v(ind)) .* ( 1 +  out_temp );
end



