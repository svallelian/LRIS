function lkhd = computeLkhd(b, A, Xchain, Muchain)
% Compute the data-misfit part of log-likelihood for given samples

    [m,~] = size(A);
    [nchains,~] = size(Muchain);
    maxIter = length(Muchain{nchains});
    B = repmat(b,[1 maxIter]);
    lkhd = cell(nchains,1);
    
    for k = 1:nchains
        X = Xchain{k};
        mu = Muchain{k};
        misfit = 0.5*sum((A*X-B).^2,1);
        lkhd{k} = mu.^(m/2).*exp(-mu.*misfit);
    end

end