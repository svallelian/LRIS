function [ y ] = computeMVPSRF(chains)
% Compute the MVPSRF of different MCMC chains (Brooks and Gelman, 1998, JCGS)
% Using the last 50% of samples in the chains
%
% y = (P-1)/P + \lambda_1*(n + 1)/n
%
% where \lambda_1 = largest eigenvalue of (W^-1)*B/P, W = within chain 
% covariance matrix, and B = between chain covariance matrix

    [nchains,~] = size(chains);
    [Nx, maxIter] = size(chains{nchains});
    P = maxIter/2;

    chainmeans = zeros(Nx, nchains);
    block = zeros(Nx, nchains*P);
    meanblock = zeros(Nx, nchains*P);
    for k = 1:nchains
        chainmeans(:,k) = mean(chains{k}(:,P+1:maxIter), 2); % sample average per chain
        block(:,(k-1)*P+1:k*P) = chains{k}(:,P+1:maxIter);
        meanblock(:,(k-1)*P+1:k*P) = repmat(chainmeans(:,k),[1 P]);
    end

    gMeanX = mean(block, 2); % overall sample average

    % Between-chain covariance
    G = repmat(gMeanX,[1 nchains]);
    B = (P/(nchains-1)).*((chainmeans - G)*(chainmeans - G)');

    % Within-chain covariance
    W = (1/(nchains*(P-1))).*((block - meanblock)*(block - meanblock)');

    % Lambda_1
    % Use a nugget if ill-conditioned
    %delta = 1e-4;
    %l1 = eig(B./P + delta*eye(Nx), W + delta*eye(Nx));
    l1 = eigs(W\(B./P),1);
   
    y = (P-1)/P + l1*(nchains+1)/nchains;

end