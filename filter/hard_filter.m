function out = hard_filter(z, lambda, v, gamma)
% v = []; gamma = [];
z = z(:); 
len = length(z);
lam = lambda(1);
out = zeros(len, 1);

out = z .* (abs(z)>=lam);
end