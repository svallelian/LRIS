function [ y ] = computePSRF(chains)
% Compute the PSRF of different MCMC chains (Brooks and Gelman, 1998, JCGS)
% Using the last 50% of samples in the chains

    [nchains,~] = size(chains);
    maxIter = length(chains{nchains});
    P = maxIter/2;

    block = zeros(nchains, P);
    for k = 1:nchains
        block(k,:) = chains{k}(P+1:maxIter);
    end

    temp = mean(block, 2); % sample average per chain
    temp2 = mean(reshape(block,[1 nchains*P])); % overall sample average
    B = (P/(nchains-1)).*(sum((temp-temp2).^2)); % between-chain variance
    W = mean(var(block, 0, 2)); % within-chain variance

    y = (P-1)/P + (nchains-1)/nchains.*(B/(P*W));


end