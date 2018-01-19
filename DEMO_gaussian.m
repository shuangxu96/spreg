N = 50; p = 100;pp=5;
X = randn(N,p);
beta = [3*ones(1,2),-3*ones(1,pp-2),zeros(1,p-pp)]'; 
bi = beta; bi(bi~=0)=1;
y = [ones(N,1),X]*[10;beta]+1*randn(N,1);

%% L2 loss + L0 penalty
% fit
model_L0 = hardreg_gauss(X, y, []);
figure(1)
subplot(1,2,1),sp_plot(model_L0)
% cv fit
cvmodel_L0 = cvhardreg_gauss(X, y, []);
subplot(1,2,2),sp_cvplot(cvmodel_L0)
%% L2 loss + L1/2 penalty
% fit
model_half = halfreg_gauss(X, y, []);
figure(2)
subplot(1,2,1),sp_plot(model_half)
% cv fit
cvmodel_half = cvhalfreg_gauss(X, y, []);
subplot(1,2,2),sp_cvplot(cvmodel_half)
%% L2 loss + L1 penalty (Lasso)
% fit
model_lasso = lassoreg_gauss(X, y, []);
figure(3)
subplot(1,2,1),sp_plot(model_lasso)
% cv fit
cvmodel_lasso = cvlassoreg_gauss(X, y, []);
subplot(1,2,2),sp_cvplot(cvmodel_lasso)
%% L2 loss + SCAD penalty (SCAD)
% fit
model_scad = scadreg_gauss(X, y, []);
figure(4)
subplot(1,2,1),sp_plot(model_scad)
% cv fit
cvmodel_scad = cvscadreg_gauss(X, y, []);
subplot(1,2,2),sp_cvplot(cvmodel_scad)
%% L2 loss + MC penalty (MCP)
% fit
model_mcp = mcpreg_gauss(X, y, []);
figure(5)
subplot(1,2,1),sp_plot(model_mcp)
% cv fit
cvmodel_mcp = cvmcpreg_gauss(X, y, []);
subplot(1,2,2),sp_cvplot(cvmodel_mcp)

%% L2 loss + L2 penalty (Ridge)
% fit
opts.gamma = 0.5;
model_ridge = enetreg_gauss(X, y, opts);
figure(6)
subplot(1,2,1),sp_plot(model_ridge)
% cv fit
cvmodel_lasso = cvlassoreg_gauss(X, y, []);
subplot(1,2,2),sp_cvplot(cvmodel_lasso)
%% L2 loss + without penalty (OLS)
% fit
model_ols = ols(X, y, []);



%% plot all traces
figure(6)
subplot(2,3,1),sp_plot(model_mcp)
subplot(2,3,2),sp_plot(model_scad)
subplot(2,3,3),sp_plot(model_lasso)
subplot(2,3,4),sp_plot(model_hard)
subplot(2,3,5),sp_plot(model_half)