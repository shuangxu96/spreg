function model = enet(X, y, opt)
%% Pre-processing
[n, p] = size(X);
model.X = X; model.y = y;
if (~isfield(opt,'maxiter')); maxiter = 1000; else maxiter = opt.maxiter; end
if (~isfield(opt,'tol')); tol = 1e-10; else tol = opt.tol; end
if (~isfield(opt,'nlambda')); nlambda = 100; else nlambda = opt.nlambda; end
if (~isfield(opt,'xtest')); xtest = []; else xtest = opt.xtest; end
if (~isfield(opt,'ytest')); ytest = []; else ytest = opt.ytest; end
if (~isfield(opt,'gamma')); gamma = 0.5; else gamma = opt.gamma; end
if gamma == 1; penalty = 'lasso'; elseif gamma == 0; penalty ='ridge'; else penalty = ['enet',num2str(gamma)];end
if (~isfield(opt,'standardize')); standardize = true; else standardize = opt.standardize; end
if (~isfield(opt,'weight'));
    weight = ones(n,1);
    v = ones(p,1);
else
    weight = opt.weight;
    v = zeros(p, 1);
    for j = 1: (p); v(j) = X(:,j)' *  (X(:,j) .* weight);end;
    v = v / n;
end
muX = mean(bsxfun(@times, X, weight));
muY = (y' * weight)/n;
y = y - muY;
if standardize
X = bsxfun(@minus,X,muX);
sigmaX = sqrt( weight'*(X.^2)/(n-1) );
sigmaX(sigmaX==0) = 1;
X = bsxfun(@rdivide, X, sigmaX);
end
if (~isfield(opt,'lambda'));
    lambda_max = max((X'*(y.*weight))./v)/n;  %mcp/scad
    lambda = logspace(log10(lambda_max)-5, log10(lambda_max), nlambda);
else
    lambda = opt.lambda;
end
nlambda = length(lambda);
%%
out_beta = zeros(p,nlambda);
out_beta0 = zeros(1,nlambda);
step = zeros(1,nlambda);
%% iteration
for nl = 1:nlambda
    beta = zeros(p,1);
    r = y - mean(y);
    t = 0;
    converge = false;
    while ~converge
        t = t + 1;
        beta_new = beta;
        
        % update
        for j = 1:p
            zj = X(:,j)' * (r .* weight) / n + v(j) * beta_new(j);
            beta_new(j) = enet_filter(zj, lambda(nl), v(j), gamma);
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
    out_beta(:,nl) = beta;
    out_beta0(1,nl) = muY-muX*beta;
    step(1,nl) = t;
end
%% output
model.SampleSize = n;
model.FeatureSize = p;
model.beta = out_beta;
model.beta0 = out_beta0;
model.step = step;
model.weight = weight;
model.lambda = lambda;
model.nlambda = nlambda;
model.Penalty = penalty;
model.Model = 'Gauss';
model.CLASS = 'spreg';
model.stat = sp_model_assess(X, y, model);
if ~isempty(xtest); model.stat_test = sp_model_assess(xtest, ytest, model); end