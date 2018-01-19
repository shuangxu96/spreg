function model = cvhard_lad(X, y, opt)
model.spreg = hard_lad(X, y, opt);
model.CLASS = 'cv_spreg';
inx = sum(abs(model.spreg.beta));
opt.lambda = model.spreg.lambda(inx ~= 0);
n = model.spreg.SampleSize;
nlambda = model.spreg.nlambda;
if (~isfield(opt,'nfold')); nfold = 10; else nfold = opt.nfold; end

cvid = 1 + mod((1:n)',nfold);
cvid = randsample(cvid, n, false);

cvMAE = zeros(nfold, nlambda);
cvAIC = zeros(nfold, nlambda);
cvBIC = zeros(nfold, nlambda);
cvEBIC = zeros(nfold, nlambda);

for nf = 1:nfold
    opt.xtest = X(cvid == nf,:);
    opt.ytest = y(cvid == nf,:);
    Xtrain = X(cvid ~= nf,:);
    ytrain = y(cvid ~= nf,:);
    
    cvmodel = hard_lad(Xtrain, ytrain, opt);
    cvMAE(nf,:) = cvmodel.stat_test.MAE;
    cvAIC(nf,:) = cvmodel.stat_test.AIC;
    cvBIC(nf,:) = cvmodel.stat_test.BIC;
    cvEBIC(nf,:) = cvmodel.stat_test.EBIC;
end
model.lambda = opt.lambda;
model.MAEm = mean(cvMAE);
model.AICm = mean(cvAIC);
model.BICm = mean(cvBIC);
model.EBICm = mean(cvEBIC);
model.MAEsd = std(cvMAE, 0, 1);
model.AICsd = std(cvAIC, 0, 1);
model.BICsd = std(cvBIC, 0, 1);
model.EBICsd = std(cvEBIC, 0, 1);
model.MAElo = model.MAEm - model.MAEsd;
model.AIClo = model.AICm - model.AICsd;
model.BIClo = model.BICm - model.BICsd;
model.EBIClo = model.EBICm - model.EBICsd;
model.MAEup = model.MAEm + model.MAEsd;
model.AICup = model.AICm + model.AICsd;
model.BICup = model.BICm + model.BICsd;
model.EBICup = model.EBICm + model.EBICsd;

%% lambda
[~,model.LambdaMinIndex] = min([model.MAEm;model.AICm;model.BICm;model.EBICm],[],2);
model.LambdaMin = model.lambda(model.LambdaMinIndex);
% mae
i1 = find((model.MAEm<model.MAEup(model.LambdaMinIndex(1))));
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