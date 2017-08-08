function [ y ] = computeCA(chains,P)
% Compute the cumulative averages of different MCMC chains (univariate)
% Using the last 50% of samples in the chains unless otherwise specified

    [nchains,~] = size(chains);
    maxIter = length(chains{nchains});

    if nargin < 2
        P = maxIter/2;
    end

    y = cell(nchains,1);
    for k = 1:nchains
        y{k} = cumsum(chains{k}(P+1:maxIter))./(1:(maxIter-P));
    end

end