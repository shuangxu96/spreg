function model = lasso_lad(X, y, opt)
%% Pre-processing
penalty = 'lasso';
[n, p] = size(X);
model.X = X; model.y = y;
if (~isfield(opt,'maxiter')); maxiter = 1000; else maxiter = opt.maxiter; end
if (~isfield(opt,'tol')); tol = 1e-10; else tol = opt.tol; end
if (~isfield(opt,'nlambda')); nlambda = 100; else nlambda = opt.nlambda; end
if (~isfield(opt,'xtest')); xtest = []; else xtest = opt.xtest; end
if (~isfield(opt,'ytest')); ytest = []; else ytest = opt.ytest; end
if (~isfield(opt,'weight'));
    weight = ones(n,1);
else
    weight = opt.weight;
end
% muX = mean(bsxfun(@times, X, weight));
% muY = (y' * weight)/n;
if (~isfield(opt,'lambda'));
    lambda_max = max( sum(abs(X)) );  %mcp/scad
    lambda = logspace(log10(lambda_max)-3, log10(lambda_max), nlambda);
else
    lambda = opt.lambda;
end
nlambda = length(lambda);
y = [y;zeros(p,1)];

%%
out_beta = zeros(p,nlambda);
out_beta0 = zeros(1,nlambda);
step = zeros(1,nlambda);
%% iteration
for nl = 1:nlambda
    beta = zeros(p+1,1);
    A = [ones(n,1),bsxfun(@times, X, weight); zeros(p,1), lambda(nl)*eye(p)];
    W_mat = abs(A);
    r = y;
    t = 0;
    converge = false;
    while ~converge
        t = t + 1;
        beta_new = beta;
        
        % update
        for j = 1:(p+1)
            s = (r + A(:,j) * beta_new(j) ) ./ A(:,j) ;
            beta_new(j) = weight_median(s, W_mat(:,j));
            r = r - (beta_new(j) - beta(j)) * A(:,j);
        end

        % converge or not
        if t >= maxiter
            converge = true;
        else
            converge = mean((beta_new - beta).^2)<tol;
        end
        beta = beta_new;
    end
    out_beta(:,nl) = beta(2:end, :);
    out_beta0(1,nl) = beta(1, :);
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
model.Model = 'Laplace';
model.CLASS = 'spreg';
model.stat = sp_model_assess(model.X, model.y, model);
if ~isempty(xtest); model.stat_test = sp_model_assess(xtest, ytest, model); end