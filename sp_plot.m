function sp_plot(model, varargin)
if isequal(model.CLASS,'cv_spreg')
    model = model.spreg;
    varargin = {};
end
    B = model.beta;
    B(~isfinite(B)) = NaN;
    
    % 'PlotType'
    pnames =   { 'PlotType' };
    [myPlotType, ~, varargin] = internal.stats.parseArgs(pnames, [], varargin{:});
    if isempty(myPlotType)
        myPlotType = 'Lambda';
    end
    plotType = internal.stats.getParamVal(myPlotType, ...
        {'L1', 'L2', 'Lambda', 'L0.5'}, 'PlotType');
    switch plotType
        case 'L1'
            xvals = nansum(abs(B),1);
            xlab = 'L1';
        case 'L2'
            xvals = sqrt(nansum((B.^2),1));
            xlab = 'L2';
        case 'Lambda'
            xvals = model.lambda;
            xlab = 'Lambda';
        case 'L0.5'
            xvals = (nansum((B.^0.5),1)).^2;
            xlab = 'L0.5';
    end
    xvals(all(isnan(B),1)) = NaN;
    ind0 = find(sum(B,1)==0);
    if length(ind0)>=5
        B(:,ind0(1:end-4))=[];
        xvals(:,ind0(1:end-4))=[];
    end
    % Plot
    processSEMILOGX(xvals, B, varargin)
%     ax = gca;
%     ax.XAxis.Direction = 'reverse';
    title(['Trace plot of coefficients by ',upper(model.Penalty)])
    xlabel(xlab)
    xlim([min(xvals),max(xvals)*1.5])
end %-sp_plot(model, varargin)

%% sub-function
function processSEMILOGX(xvals, B, vain)
len = length(vain);
switch len
    case 0
        semilogx(xvals, B')
    case 2
        semilogx(xvals, B',  vain{1},vain{2})
    case 4
        semilogx(xvals, B',  vain{1},vain{2},vain{3},vain{4})
    case 6
        semilogx(xvals, B',  vain{1},vain{2},vain{3},vain{4},...
            vain{5},vain{6})
    case 8
        semilogx(xvals, B',  vain{1},vain{2},vain{3},vain{4},...
            vain{5},vain{6},vain{7},vain{8})
    otherwise
        semilogx(xvals, B',  vain{1},vain{2},vain{3},vain{4},...
            vain{5},vain{6},vain{7},vain{8},vain{9},vain{10})
end
end %-processSEMILOGX(xvals, B, vain)