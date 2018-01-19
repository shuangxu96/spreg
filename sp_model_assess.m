function out = sp_model_assess(X, y, model)
n = model.SampleSize;
p = model.FeatureSize;
out.yfitted = sp_predict(X, model);
out.p1 = sum(model.beta~=0);
switch model.Model
    case 'Gauss'
        out.MSE = mean(bsxfun(@minus,y,out.yfitted).^2);
        out.EBIC = n * log(out.MSE + 1e-10) + (log(n) + 2 * log(p)) * out.p1 / n;
        out.BIC = n * log(out.MSE + 1e-10) + out.p1 * log(n);
        out.AIC = n * log(out.MSE + 1e-10) + out.p1 * 2;
    case 'Laplace'
        out.MAE = mean(bsxfun(@minus,y,out.yfitted).^2);%mean(abs(bsxfun(@minus,y,out.yfitted)));
        out.EBIC = n * log(out.MAE + 1e-10) + (log(n) + 2 * log(p)) * out.p1 / n;
        out.BIC = n * log(out.MAE + 1e-10) + out.p1 * log(n);
        out.AIC = n * log(out.MAE + 1e-10) + out.p1 * 2;
end
        