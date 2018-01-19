function yhat = sp_predict(X, model)
beta = model.beta;
beta0 = model.beta0;
X1 = X;
X1(:,end+1) = 1;
yhat = X1*[beta;beta0];
