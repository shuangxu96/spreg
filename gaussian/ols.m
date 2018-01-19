function model = ols(X, y, opt)
%% Pre-processing
penalty = 'without penalty';
[n, p] = size(X);
model.X = X; model.y = y;
model.SampleSize = n; model.FeatureSize = p;
if (~isfield(opt,'maxiter')); maxiter = 1000; else; maxiter = opt.maxiter; end
if (~isfield(opt,'tol')); tol = 1e-10; else; tol = opt.tol; end
if (~isfield(opt,'xtest')); xtest = []; else; xtest = opt.xtest; end
if (~isfield(opt,'ytest')); ytest = []; else; ytest = opt.ytest; end
if (~isfield(opt,'intercept')); intercept = true; else; intercept = opt.intercept; end
if (~isfield(opt,'standardize')); standardize = true; else; standardize = opt.standardize; end
if (~isfield(opt,'weight')); weight = ones(n,1); else; weight = opt.weight;end
muX = mean(bsxfun(@times, X, weight));
% muY = (y' * weight)/n;
% y = y - muY;
if standardize
    X = bsxfun(@minus,X,muX);
    sigmaX = sqrt( weight'*(X.^2)/n );
    sigmaX(sigmaX==0) = 1;
    X = bsxfun(@rdivide, X, sigmaX);
end
if intercept
    X = [ones(n,1), X];
    p = p+1;
end
v = zeros(p, 1);
for j = 1:p; v(j) = X(:,j)' *  (X(:,j) .* weight);end
v = v / n;
%%
% out_beta = zeros(p,1);
% out_beta0 = zeros(1,1);
% step = zeros(1,1);
%% iteration

beta = zeros(p,1);
r =  y;
t = 0;
converge = false;
while ~converge
    t = t + 1;
    beta_new = beta;
    
    % update
    for j = 1:p
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
if intercept
    out_beta(:,1) = beta(2:end);
    out_beta0(1,1) = beta(1);
else
    out_beta(:,1) = beta;
end
step(1,1) = t;

%% output
model.beta = out_beta;
if intercept; model.beta0 = out_beta0; end
model.step = step;
model.weight = weight;
model.v = v;
model.Penalty = penalty;
model.Model = 'Gauss';
model.CLASS = 'spreg';
model.intercept = intercept;
model.stat = sp_model_assess(model.X, y, model);
if ~isempty(xtest); model.stat_test = sp_model_assess(xtest, ytest, model); end