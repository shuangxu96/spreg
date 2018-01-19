function model = half(X, y, opt)
%%  model = halfreg_gaussian(X, y, opt)
% Fit a sparse linear model with L2 loss and L1/2 penalty.
% 
% *Inputs* :
% X - observation matrix, [n x p].
% y - output vector, containing either 1 or -1, [n x 1].
% opt.maxiter - the maximum number of iteration. Default: 1000.
% opt.tol - tolerance of error. Default: 1e-10.
% opt.lambda - penalty strength. Defualt: code automatically determines
% based on input data (X and y).
% opt.nlambda - the number of candidate lambda. If you have specified the
% value of lambda, nlambda is ineffective. Default: 100. d
% opt.weight - the weight of observations. Default: 1 for all samples.
% opt.xtest - test data. Default: [];
% opt.ytest - test data. Default: [];
%
% *Output* :
% model - the model of 'spreg'.
% model.X, model.y - the input data.
% model.SampleSize, model.FeatureSize - the dimension of model.X.
% model.beta - the estimate of coefficients.
% model.beta0 - the estimate of intercept.
% model.step - the iteration step of each lambda.
% model.weight - the weight of each sample.
% model.lambda - the candidate lambda.
% model.nlambda - the number of lambda.
% model.Penalty - 'half' (i.e. the L 1/2 norm).
% model.Model - 'Gauss' (i.e. L2 norm loss).
% model.CLASS - 'spreg'.
% model.stat - model information for train data {model.X,model.y}.
% model.stat.yfitted - the estimate of model.y (i.e. \hat{y}).
% model.stat.MSE - the mean sqaured error.
% model.stat.p1 - the number of non-zeros' coefficients.
% model.stat.EBIC - EBIC.
% model.stat.BIC - BIC.
% model.stat.AIC - AIC.
% model.stat_test - information for test data {model.xtest,model.ytest}. 
% 
% Example(s): 
% model = halfreg_gauss(X, y, []);
%
% Ref(s): 
% Xu Z, et al. L1/2 regularization: a thresholding representation theory
% and a fast solver[J]. IEEE TNNLS, 2012, 23(7):1013.  
%
% Copyright (c) 2017, Shuang Xu
% Email: xu.s@outlook.com ; shuangxu@stu.xjtu.edu.cn
% All rights reserved.
% See the file LICENSE for licensing information.

%% Pre-processing
penalty = 'half';
[n, p] = size(X); 
model.X = X; model.y = y;
if (~isfield(opt,'maxiter')); maxiter = 1000; else maxiter = opt.maxiter; end
if (~isfield(opt,'tol')); tol = 1e-10; else tol = opt.tol; end
if (~isfield(opt,'nlambda')); nlambda = 100; else nlambda = opt.nlambda; end
if (~isfield(opt,'xtest')); xtest = []; else xtest = opt.xtest; end
if (~isfield(opt,'ytest')); ytest = []; else ytest = opt.ytest; end
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
X = bsxfun(@minus,X,muX);
sigmaX = sqrt( weight'*(X.^2)/(n-1) );
sigmaX(sigmaX==0) = 1;
X = bsxfun(@rdivide, X, sigmaX);

if (~isfield(opt,'lambda'));
    z = X'*(y.*weight);  lambda_max = sqrt( max( 4*abs(z).^3 ./ (3*v)) /n);
    lambda = logspace(log10(lambda_max)-8, log10(lambda_max), nlambda);
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
            beta_new(j) = half_filter(zj, lambda(nl), v(j));
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