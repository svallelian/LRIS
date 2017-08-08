function [X, Mu, Sigma, rate] = metwithingibbsNCP(b, A, L, V, l, a_mu, b_mu, ...
                                    a_sigma, b_sigma, mu, sigma, ...
                                    zInit, maxiter)                              
% LRIS in Metropolis-Hastings-within-Gibbs sampling with non-centered parameterization 
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
%   zInit: initial value for z 
%   maxiter: # sampling iterations
%
% Outputs
%   X: samples of x
%   Mu: samples of mu
%   Sigma: samples of sigma
%   rate: overall acceptance rate
%
% D. Andrew Brown, Arvind Saibaba, Sarah Vallelian 07/2017
    
    [m,n] = size(A);
    
    % Store iteration history
    X = zeros(n,maxiter);
    Mu = zeros(1,maxiter);
    Sigma = zeros(1,maxiter);

    x = zInit; % Denoted z in the Supplementary Materials
    Atb = A'*b;
    AtA = A'*A;
   
    accept = 0;  % Track the acceptance rate for the Metropolis proposals

    for i = 1:maxiter
        
        % Sample for z from approximate normal distribution (LRIS)
        xStar = postsampleNCP(Atb,L,V,l,mu,sigma,randn(n,1));
        
        % Evaluate the Hastings ratio, note the parameterization
        rat = metropolisratio(x, xStar, A, L, V, l, mu/sigma, 1);
        
        % Accept/Reject
        u = rand(1);
        if rat >= u
            x = xStar;
            accept = accept + 1;
        end
        
        eta = (sigma).^(-0.5);
        misfit = 0.5*norm(eta.*A*x-b).^2;
        
        % Sample for mu
        mu = gamrnd(m/2 + a_mu, (misfit + b_mu).^(-1));
        
        % Sample for sigma (NCP) - indep. MH w/ truncated Gaussian proposal
        % See Agapiou et al (SIAM/ASA JUQ, 2014)
        etaVar = (mu.*x'*AtA*x)^(-1);
        etaMn = mu.*x'*Atb*etaVar;
        etaStar = etaMn + sqrt(etaVar).*randn(1);
        
        % Positivity
        while etaStar <= 0
            etaStar = etaMn + sqrt(etaVar).*randn(1);
        end
        
        % Evaluate the Hastings ratio
        num = (1/etaStar^2)^(a_sigma + 0.5).*exp(-b_sigma/etaStar^2);
        denom = (1/eta^2)^(a_sigma + 0.5).*exp(-b_sigma/eta^2);
        rat = num/denom;
        
        % Accept/Reject
        u = rand(1);
        if rat >= u
            sigma = 1/etaStar^2;
        end
        
        % Store entries
        X(:,i) = eta.*x;  % Rescale from z to x
        Mu(i)  = mu;
        Sigma(i) = sigma;
        
   end
  
   rate = accept./maxiter;

end