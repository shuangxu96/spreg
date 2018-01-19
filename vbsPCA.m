function vbsPCA(X,r)
alpha0 = 1e-2;
beta0 = 1e-4;
eta_b = 1e-3;
eta_w = 1e-3;
eta_0 = 1e+3;


Etau = alpha0/beta0;
mu_b = mean(X)';
[N,p] = size(X);
X2 = 0;
for i=1:N
    X2 = X2 + X(i,:)'*X(i,:);
end
Xc = bsxfun(@minus,X,mu_b');
[outU,outS,outV] = initial_vbsPCA(Xc, r);
mu_W = outV;
for l = 1:p; sigma_W(:,:,l) = mu_W(l,:)'*mu_W(l,:);end
mu_H = outU*outS;
Theta = double(mu_W~=0);
alpha = alpha0 + N*p/2;
beta = beta0 +  X2
Etau = alpha/beta;



% h
sigma_H = inv(eye(r) + Etau * Eww);
mu_H = Etau * bsxfun(@minus, X, mu_b') * mu_W  * sigma_H';
% b
sigma_b = eye(p)/(N*Etau + eta_b);
mu_b = Etau * sigma_b * sum(X - mu_H*mu_W')';
Xc = bsxfun(@minus, X, mu_b');
% W
eta = Theta*eta_w + (1-Theta)*eta_0;
for l = 1:p
    sigma_W(:,:,l) = inv(Etau * (N*sigma_H + mu_H'*mu_H) + diag(eta(l,:)));
    mu_W(l,:) = Etau * Xc(:,l)' * mu_H * sigma_W(:,:,l)';
end
Eww = sum(sigma_W,3) + mu_W'*mu_W );
% gamma
Ew2 = zeros(p,r);
for i = 1:p
    Ew2(i,:) = diag(sigma_W(:,:,i))';
end
Ew2 = mu_W.^2 + Ew2;
Theta = log( rho/(1-rho) ) + Ew2*(eta_0-eta_w)/2;
Theta = 1./(1+exp(-Theta));
% tau
beta = 

logtheta = 