
maxiter = 100;
tol = 1e-4;

lad_fit = lm_lap(X, y, []);
beta = [lad_fit.beta0; lad_fit.beta];
r = y - lad_fit.stat.yfitted;
sigma = median(abs(r - median(r)))/0.675;
% C = null(X')';
% yc = C*y;
lam2 = 5;
J_old = sum((r - s).^2) + lam2*sigma*length(y);
converge = false;
step = 1;
while ~converge
    % update s
    s = r .* (abs(r) > lam2*sigma);
    % update beta
    ys = y - s;
    lad_fit = lm_lap(X, ys, []);
    beta = [lad_fit.beta0; lad_fit.beta];
    r = y - lad_fit.stat.yfitted;
    % update converge
    J_new = sum((r - s).^2) + lam2*sigma*sum(s~=0);
    if step > maxiter
        converge = true;
    elseif abs(J_new - J_old) < tol 
        converge = true;
    else
        converge = false;
    end
    J_old = J_new;
    step = step + 1;
end

X = rand(100,10);
betaT = 2*rand(10,1);
r = [0.5*randn(90,1); 10*ones(10,1)];
y = X*betaT+r;