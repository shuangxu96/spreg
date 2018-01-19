function v = weight_median(seq,wei)
% To find the weighted median of a sequence.
% seq: The sequence of real number.
% wei: The non-negative weights
seq = seq(:);
wei = wei(:);
len = length(seq);
if len == 1
    v = seq;
elseif len == 2
    if seq(1) >= seq(2)
        v = seq(1);
    else
        v = seq(2);
    end
else
    [seq_sort, IND] = sort(seq);
    wei_sort = wei(IND);
    wei_cumsum = cumsum(wei_sort);
    wei_half = sum(wei)/2;
    opt_ind = max(find(wei_cumsum < wei_half));
    if isempty(opt_ind)
        opt_ind = 1;
    end
    if opt_ind < len
        opt_ind = opt_ind+1;
    end
    v = seq_sort(opt_ind);
end
v = v(1);