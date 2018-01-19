cvmodel = cvhalfreg_gaussian(X, y,[] );
sp_cvplot(cvmodel,'CVType','MSE')
sp_plot(cvmodel,'CVType','MSE')

model = halfreg_gaussian(X, y,[] );
sp_cvplot(model)
sp_plot(model,'PlotType','Lambda')

