function [X, Mu, Sigma, lkhd] = blockgibbs(b, A, L, a_mu, b_mu, ...
                                    a_sigma, b_sigma, mu, sigma, maxiter)
% Block Gibbs sampling with conjugate Gamma priors on mu and sigma
%
% Inputs
%   b: data vector
%   A: forward operator
%   L: Cholesky factor of prior covariance matrix
%   a_mu, b_mu: shape/rate hyperparameters on precision mu
%   a_sigma, b_sigma: shape/rate hyperparameters on precision sigma
%   mu: initial value for mu
%   sigma: initial value for sigma
%   maxiter: # sampling iterations
%
% Outputs
%   X: samples of x
%   Mu: samples of mu
%   Sigma: samples of sigma
%   lkhd: data-misfit part of log-likelihood
%
% D. Andrew Brown, Arvind Saibaba, Sarah Vallelian 07/2017
    
    [m,n] = size(A); 
    
    %Store iteration history
    X = zeros(n,maxiter);
    Mu = zeros(1,maxiter);
    Sigma = zeros(1,maxiter);
    lkhd = zeros(1,maxiter);
    
    Atb = A'*b;
    AtA = A'*A;
    LtL = L'*L;

   for i = 1:maxiter
       
        %Sample for x from normal distribution
        R = chol(mu*AtA + sigma*LtL);
        x = R\(R'\(mu*Atb) + randn(n,1));   
        
        misfit = 0.5*norm(A*x-b).^2;
        reg = 0.5*norm(L*x).^2;
        
        %Sample for mu
        mu = gamrnd(m/2 + a_mu, (misfit + b_mu).^(-1));
        
        %Sample for sigma
        sigma = gamrnd(n/2 + a_sigma, (reg + b_sigma).^(-1));
        
        %Store entries
        X(:,i) = x;
        Mu(i)  = mu;
        Sigma(i) = sigma;
        lkhd(i) = mu^(m/2)*exp(-mu*misfit);
        
    end    

end