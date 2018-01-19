function model = cvscad(X, y, opt)
model.spreg = scad(X, y, opt);
model.CLASS = 'cv_spreg';
inx = sum(abs(model.spreg.beta));
opt.lambda = model.spreg.lambda(inx ~= 0);
n = model.spreg.SampleSize;
nlambda = length(opt.lambda);
if (~isfield(opt,'nfold')); nfold = 5; else; nfold = opt.nfold; end

cvid = 1 + mod((1:n)',nfold);
cvid = randsample(cvid, n, false);

cvMSE = zeros(nfold, nlambda);
cvAIC = zeros(nfold, nlambda);
cvBIC = zeros(nfold, nlambda);
cvEBIC = zeros(nfold, nlambda);

for nf = 1:nfold
    opt.xtest = X(cvid == nf,:);
    opt.ytest = y(cvid == nf,:);
    Xtrain = X(cvid ~= nf,:);
    ytrain = y(cvid ~= nf,:);
    
    cvmodel = scad(Xtrain, ytrain, opt);
    cvMSE(nf,:) = cvmodel.stat_test.MSE;
    cvAIC(nf,:) = cvmodel.stat_test.AIC;
    cvBIC(nf,:) = cvmodel.stat_test.BIC;
    cvEBIC(nf,:) = cvmodel.stat_test.EBIC;
end
model.lambda = opt.lambda;
model.MSEm = mean(cvMSE);
model.AICm = mean(cvAIC);
model.BICm = mean(cvBIC);
model.EBICm = mean(cvEBIC);
model.MSEsd = std(cvMSE, 0, 1);
model.AICsd = std(cvAIC, 0, 1);
model.BICsd = std(cvBIC, 0, 1);
model.EBICsd = std(cvEBIC, 0, 1);
model.MSElo = model.MSEm - model.MSEsd;
model.AIClo = model.AICm - model.AICsd;
model.BIClo = model.BICm - model.BICsd;
model.EBIClo = model.EBICm - model.EBICsd;
model.MSEup = model.MSEm + model.MSEsd;
model.AICup = model.AICm + model.AICsd;
model.BICup = model.BICm + model.BICsd;
model.EBICup = model.EBICm + model.EBICsd;

%% lambda
[~,model.LambdaMinIndex] = min([model.MSEm;model.AICm;model.BICm;model.EBICm],[],2);
model.LambdaMin = model.lambda(model.LambdaMinIndex);
% mse
i1 = find((model.MSEm<model.MSEup(model.LambdaMinIndex(1))));
[model.Lambda1SE(1),i2] = max(model.lambda(i1));
model.Lambda1SEIndex(1) = i1(i2);
% aic
i1 = find((model.AICm<model.AICup(model.LambdaMinIndex(2))));
[model.Lambda1SE(2),i2] = max(model.lambda(i1));
model.Lambda1SEIndex(2) = i1(i2);
% bic
i1 = find((model.BICm<model.BICup(model.LambdaMinIndex(3))));
[model.Lambda1SE(3),i2] = max(model.lambda(i1));
model.Lambda1SEIndex(3) = i1(i2);
% ebic
i1 = find((model.EBICm<model.EBICup(model.LambdaMinIndex(4))));
[model.Lambda1SE(4),i2] = max(model.lambda(i1));
model.Lambda1SEIndex(4) = i1(i2);