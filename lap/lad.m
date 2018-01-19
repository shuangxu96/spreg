function model = lad(X, y, opt)
%% Pre-processing
penalty = 'without penalty';
[n, p] = size(X);
model.X = X; model.y = y;
if (~isfield(opt,'maxiter')); maxiter = 1000; else maxiter = opt.maxiter; end
if (~isfield(opt,'tol')); tol = 1e-10; else tol = opt.tol; end
if (~isfield(opt,'xtest')); xtest = []; else xtest = opt.xtest; end
if (~isfield(opt,'ytest')); ytest = []; else ytest = opt.ytest; end
if (~isfield(opt,'weight'));
    weight = ones(n,1);
else
    weight = opt.weight;
end
muX = mean(bsxfun(@times, X, weight));
muY = (y' * weight)/n;
% y = y - muY;
% X = bsxfun(@minus,X,muX);
% sigmaX = sqrt( weight'*(X.^2)/(n-1) );
% sigmaX(sigmaX==0) = 1;
% X = bsxfun(@rdivide, X, sigmaX);
A = [ones(n,1),X];
W_mat = bsxfun(@times, abs(A), weight);
%%
out_beta = zeros(p,1);
out_beta0 = zeros(1,1);
step = zeros(1,1);
%% iteration

    beta = zeros(p+1,1);
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
    out_beta(:,1) = beta(2:end);
    out_beta0(1,1) = beta(1);
    step(1,1) = t;

%% output
model.SampleSize = n;
model.FeatureSize = p;
model.beta = out_beta;
model.beta0 = out_beta0;
model.step = step;
model.weight = weight;
model.Penalty = penalty;
model.Model = 'Laplace';
model.CLASS = 'spreg';
model.stat = sp_model_assess(X, y, model);
if ~isempty(xtest); model.stat_test = sp_model_assess(xtest, ytest, model); end