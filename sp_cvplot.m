function sp_cvplot(model, varargin)
if isequal(model.CLASS,'spreg')
    sp_plot(model)
else
    xvals = model.lambda;
    % CVType
    pnames =   { 'CVType' };
    [myCVType, ~, varargin] = internal.stats.parseArgs(pnames, [], varargin{:});
    if isempty(myCVType)
        myCVType = 'EBIC';
    end
    CVType = internal.stats.getParamVal(myCVType, ...
        {'EBIC', 'BIC', 'AIC', 'MSE'}, 'CVType');
    switch CVType
        case 'EBIC'
            yvals = model.EBICm;
            E = model.EBICsd;
            lmin = model.LambdaMin(4);
            l1se = model.Lambda1SE(4);
            ylab = 'EBIC';
        case 'BIC'
            yvals = model.BICm;
            E = model.BICsd;
            lmin = model.LambdaMin(3);
            l1se = model.Lambda1SE(3);
            ylab = 'BIC';
        case 'AIC'
            yvals = model.AICm;
            E = model.AICsd;
            lmin = model.LambdaMin(2);
            l1se = model.Lambda1SE(2);
            ylab = 'AIC';
        case 'MSE'
            yvals = model.MSEm;
            E = model.MSEsd;
            lmin = model.LambdaMin(1);
            l1se = model.Lambda1SE(1);
            ylab = 'MSE';
    end
    processERRORBAR(xvals, yvals, E, varargin)
    ylabel(ylab)
    xlabel('Lambda')
    title(['Cross-validation curve by ',upper(model.spreg.Penalty)])
    xlim([min(xvals)*0.5,max(xvals)*1.5])
    ax = gca;
    ax.XScale = 'log';
    hold on
    plot([lmin,lmin], get(gca, 'YLim'), '--', 'LineWidth', 1)
    plot([l1se,l1se], get(gca, 'YLim'), '-', 'LineWidth', 1)
    
end
end %-sp_cvplot(model, varargin)
%% sub-function
function processERRORBAR(xvals, yvals, E, vain)
len = length(vain);
switch len
    case 0
        errorbar(xvals, yvals, E, 'Color', 0.6*[1,1,1], 'Marker', '.',...
            'MarkerSize',5,'MarkerEdgeColor','red','MarkerFaceColor','red',...
            'LineStyle','none')
    case 2
        errorbar(xvals, yvals, L, U,  vain{1},vain{2})
    case 4
        errorbar(xvals, yvals, L, U,  vain{1},vain{2},vain{3},vain{4})
    case 6
        errorbar(xvals, yvals, L, U,  vain{1},vain{2},vain{3},vain{4},...
            vain{5},vain{6})
    case 8
        errorbar(xvals, yvals, L, U,  vain{1},vain{2},vain{3},vain{4},...
            vain{5},vain{6},vain{7},vain{8})
    otherwise
        errorbar(xvals, yvals, L, U,  vain{1},vain{2},vain{3},vain{4},...
            vain{5},vain{6},vain{7},vain{8},vain{9},vain{10})
end
end %-processERRORBAR(xvals, yvals, E, vain)