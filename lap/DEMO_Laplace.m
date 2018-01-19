N = 50; p = 20;pp=5;
X = randn(N,p);
beta = [3*ones(1,2),-3*ones(1,pp-2),zeros(1,p-pp)]'; 
bi = beta; bi(bi~=0)=1;
y = [ones(N,1),X]*[0.5;beta]+1*random('t',10,N,1);

%% L1 loss + without penalty
% fit
model_lad = lmreg_lap(X, y, []);
figure(1)
subplot(1,2,1),sp_plot(model_lad)

%% L1 loss + lasso penalty
% fit
model_lad_lasso = lasso_lap(X, y, []);
figure(2)
subplot(1,2,1),sp_plot(model_lad_lasso)


%% plot all traces
figure(6)
subplot(2,3,1),sp_plot(model_mcp)
subplot(2,3,2),sp_plot(model_scad)
subplot(2,3,3),sp_plot(model_lasso)
subplot(2,3,4),sp_plot(model_hard)
subplot(2,3,5),sp_plot(model_half)


x=-1.5:0.01:1.5;
y = abs(x).^(0.5);
plot(x,y)