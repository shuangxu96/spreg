function out = half_phi(z,lam,v)
if ~isempty(z)
    z = (abs(z) / 3).^(-1.5);
    z(z>1) = 1;
    z(z<-1) = -1;
    out = acos(lam * z .* sqrt(v) / 8);
else
    out = [];
end
end