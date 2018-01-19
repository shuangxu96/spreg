N = 1000; p = 50;pp=5;
X = randn(N,p);
beta = [3*ones(1,2),-3*ones(1,pp-2),zeros(1,p-pp)]'; 
bi = beta; bi(bi~=0)=1;
y = logsig([ones(N,1),X]*[10;beta]);%+0.5*randn(N,1));
y = random('bino',1,y);

model = logreg3(X, y, []);

model = lassoreg_logit(X, y, []);

[B,dev,stats] = mnrfit(X, categorical(~logical(y)));

scatter([2;beta], model.beta)
figure
scatter([2;beta], B)