function [ y ] = thinning(chains, space)
% Thin the given Markov chain with given spacing

    [nchains,~] = size(chains);
    [~, maxIter] = size(chains{nchains});

    if space >= maxIter
        fprintf('Error: space must be less than chain length. \n');
        return
    end

    y = cell(nchains,1);
    for k = 1:nchains
        y{k} = chains{k}(:,1:space:maxIter);
    end

end