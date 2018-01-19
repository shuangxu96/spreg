function model = logreg(X, y, opt)
%% Pre-processing
penalty = 'without penalty';
[n, p] = size(X);
model.X = X; model.y = y;
if (~isfield(opt,'maxiter')); maxiter = 100; else maxiter = opt.maxiter; end
if (~isfield(opt,'tol')); tol = 1e-10; else tol = opt.tol; end
if (~isfield(opt,'xtest')); xtest = []; else xtest = opt.xtest; end
if (~isfield(opt,'ytest')); ytest = []; else ytest = opt.ytest; end

%% Initial iteration with beta = 0
beta = zeros(p,1);
weight = ones(n, 1) * 0.25;
y_fit = 4 * y - 2 ;
opt_wols.maxiter = 100;
opt_wols.standardize = false;
opt_wols.weight = weight;
wols = ols(X, y_fit, opt_wols);
beta = [wols.beta0; wols.beta];

%% Iteration
converge = false; t = 1;
while ~converge
    t = t + 1;
    Xb = [ones(n,1), X] * beta;
    mu = 1 ./ (1 + exp (- Xb) );
    mu(mu==0) = 1e-5; mu(mu==1) = 0.99999;
     weight = mu .* (1 - mu);
    y_fit = Xb + (y - mu) ./ weight;
    opt_wols.weight = weight;
    wols = ols_for_logit(X, y_fit, opt_wols);
    beta_new = [wols.beta0; wols.beta];
    
    if t >= maxiter
        converge = true;
    else
        converge = mean((beta_new - beta).^2)<tol;
    end
    beta = beta_new;
end
step = t;
%% output
model.SampleSize = n;
model.FeatureSize = p;
model.beta = beta(2:end);
model.beta0 = beta(1);
model.step = step;
model.weight = weight;
model.Penalty = penalty;
model.Model = 'LogisticReg';
model.CLASS = 'spreg';
% model.stat = sp_model_assess(X, y, model);
if ~isempty(xtest); model.stat_test = sp_model_assess(xtest, ytest, model); end
end


function model = ols_for_logit(X, y, opt)
%% Pre-processing
[n, p] = size(X);
model.X = X; model.y = y;
if (~isfield(opt,'maxiter')); maxiter = 100; else maxiter = opt.maxiter; end
if (~isfield(opt,'tol')); tol = 1e-10; else tol = opt.tol; end

weight = opt.weight;
v = zeros(p, 1);
for j = 1: (p); v(j) = X(:,j)' *  (X(:,j) .* weight);end;
v = v / n;

muX = mean(bsxfun(@times, X, weight));
muY = (y' * weight)/n;

%
out_beta = zeros(p,1);
out_beta0 = zeros(1,1);
step = zeros(1,1);
% iteration

 beta = zeros(p,1);
r = y - muY;
t = 0;
converge = false;
while ~converge
    t = t + 1;
    beta_new = beta;
    
    % update
    for j = 1:(p)
        zj = X(:,j)' * (r .* weight) / n + v(j) * beta_new(j);
        beta_new(j) = zj / v(j);
        r = r - (beta_new(j) - beta(j)) * X(:,j);
    end
    
    % converge or not
    if t >= maxiter
        converge = true;
    else
        converge = mean((beta_new - beta).^2)<tol;
    end
    beta = beta_new;
end
out_beta(:,1) = beta;
% out_beta(:,1) = beta(2:end);
out_beta0(1,1) = muY - muX*beta;
% out_beta0(:,1) = beta(1);
step(1,1) = t;

% output
model.SampleSize = n;
model.FeatureSize = p;
model.beta = out_beta;
model.beta0 = out_beta0;
model.step = step;
model.weight = weight;
end