function [X, Mu, Sigma, rats, rate] = metwithingibbsLRIS(b, A, L, V, l, a_mu, b_mu, ...
                                    a_sigma, b_sigma, mu, sigma, ...
                                    xInit, maxiter)                                
% LRIS in Metropolis-Hastings-within-Gibbs sampling with conjugate Gamma priors on mu and sigma
%
% Inputs
%   b: data vector
%   A: forward operator
%   L: Cholesky factor of prior covariance matrix
%   V, l: Approximate eigendecomposition of prior preconditioned Hessian
%   a_mu, b_mu: shape/rate hyperparameters on precision mu
%   a_sigma, b_sigma: shape/rate hyperparameters on precision sigma
%   mu: initial value for mu
%   sigma: initial value for sigma
%   xInit: initial value for x
%   maxiter: # sampling iterations
%
% Outputs
%   X: samples of x
%   Mu: samples of mu
%   Sigma: samples of sigma
%   rats: Metropolis-Hastings ratios per sample
%   rate: overall acceptance rate
%
% D. Andrew Brown, Arvind Saibaba, Sarah Vallelian 07/2017

    [m,n] = size(A);

    % Store iteration history
    X = zeros(n,maxiter);
    Mu = zeros(1,maxiter);
    Sigma = zeros(1,maxiter);
    rats = zeros(1,maxiter);

    x = xInit;
    Atb = A'*b;

    accept = 0;  % Track the acceptance rate for the Metropolis proposals

    for i = 1:maxiter
       
        % Sample for x from approximate normal distribution (LRIS)
        xStar = postsample(Atb,L,V,l,mu,sigma,randn(n,1));

        % Evaluate the Hastings ratio
        rat = metropolisratio(x, xStar, A, L, V, l, mu, sigma);
        rats(i) = rat;
        
        % Accept/Reject
        u = rand(1);
        if rat >= u
            x = xStar;
            accept = accept + 1;
        end
        
        misfit = 0.5*norm(A*x-b).^2;
        reg = 0.5*norm(L*x).^2;
        
        %Sample for mu
        mu = gamrnd(m/2 + a_mu, (misfit + b_mu).^(-1));
        
        %Sample for sigma
        sigma = gamrnd(n/2 + a_sigma, (reg + b_sigma).^(-1));
        
        %Store entries
        X(:,i) = x;
        Mu(i) = mu;
        Sigma(i) = sigma;
        
    end

    rate = accept./maxiter;

end