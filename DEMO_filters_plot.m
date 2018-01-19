z = -4:0.01:4;
lambda = 0.8;
flasso=lasso_filter(z,lambda);
fmcp=mcp_filter(z,lambda,[],4);
fscad=scad_filter(z,lambda,[],4.7);
fhalf=half_filter(z,lambda);
fhard = hard_filter(z, lambda);
fenet = enet_filter(z, lambda, [], 0.5);
fridge = enet_filter(z, lambda, [], 0);
% plot1
subplot(1,4,1),hold on,
plot(z,z, 'LineStyle','--', 'LineWidth',2)
plot(z,flasso, 'LineWidth',2),
plot(z,fhard, 'LineWidth',2),
legend({'Truth','L1/Lasso','L0/Hard'},'Location','Northwest')
% plot2
subplot(1,4,2),hold on,
plot(z,z, 'LineStyle','--', 'LineWidth',2)
plot(z,fmcp, 'LineWidth',2)
plot(z,fscad, 'LineWidth',2)
legend({'Truth','MCP','SCAD'},'Location','Northwest')
% plot4
subplot(1,4,3),hold on,
plot(z,z, 'LineStyle','--', 'LineWidth',2)
plot(z,fhalf, 'LineWidth',2)
legend({'Truth', 'L0.5/Half'},'Location','Northwest')
% plot4
subplot(1,4,4),hold on,
plot(z,z, 'LineStyle','--', 'LineWidth',2)
plot(z,fenet, 'LineWidth',2)
plot(z,fridge, 'LineWidth',2)
plot(z,flasso, 'LineWidth',2)
legend({'Truth', 'ElasticNet','Ridge','Lasso'},'Location','Northwest')