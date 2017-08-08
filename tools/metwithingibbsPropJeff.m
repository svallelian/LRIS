function [X, Kappa2, Upsilon, Lpost, rate] = metwithingibbsPropJeff(b, A, L, V, l, kap2Init, ...
                                upsInit, xInit, maxiter)
% LRIS in Metropolis-Hastings-within-Gibbs sampling with proper Jeffreys priors on kappa^2 and upsilon
%
% Inputs
%   b: data vector
%   A: forward operator
%   L: Cholesky factor of prior covariance matrix
%   V, l: Approximate eigendecomposition of prior preconditioned Hessian
%   kap2Init: initial value for kappa^2
%   upsInit: initial value for upsilon
%   xInit: initial value for x
%   maxiter: # sampling iterations
%
% Outputs
%   X: samples of x
%   Kappa2: samples of kappa^2
%   Upsilon: samples of upsilon
%   Lpost: logarithm of the joint posterior at sampled points
%   rate: overall acceptance rate
%
% D. Andrew Brown, Arvind Saibaba, Sarah Vallelian 07/2017

    [m,n] = size(A);

    % Store iteration history
    X = zeros(n,maxiter);
    Kappa2 = zeros(1,maxiter);
    Upsilon = zeros(1,maxiter);
    Lpost = zeros(1,maxiter);

    kap2 = kap2Init;
    ups = upsInit;
    x = xInit;
    Atb = A'*b;
   
    accept = 0;  % Track the acceptance rate for the Metropolis proposals

    for i = 1:maxiter
        
        % Sample for x from approximate normal distribution (LRIS)
        % Note the parameterization of precision components in terms of
        % variance components
        xStar = postsample(Atb,L,V,l, 1./kap2,1./(kap2.*ups),randn(n,1));
        
        % Evaluate the Hastings ratio
        rat = metropolisratio(x, xStar, A, L, V, l, 1./kap2, 1./(kap2.*ups));

        % Accept/reject
        u = rand(1);
        if rat >= u
            x = xStar;
            accept = accept + 1;
        end
        
        misfit = 0.5*norm(A*x-b).^2;
        reg = 0.5*norm(L*x).^2;

        % Sample for kappa^2 based on Jeffreys prior
        shape1 = (m+n)./2;
        shape2 = (misfit + reg./ups).^(-1);
        temp1 = gamrnd(shape1, shape2);
        kap2 = 1./temp1;
        
        % Sample for upsilon using independent MH sampling step
        shapeA = n/2+1;
        shapeB = kap2.*reg.^(-1);
        temp2 = gamrnd(shapeA, shapeB);
        etaStar = 1./temp2;
        
        rat = (etaStar.^2*(1 + ups).^2)/( ups.^2*(1 + etaStar).^2 );
        
        u = rand(1);
        if rat >= u
            ups = etaStar;
        end
        
        % Store entries
        X(:,i) = x;
        Kappa2(i) = kap2;
        Upsilon(i) = ups;
        Lpost(i) = (-1.*(m+n)/2 - 1).*log(kap2) - misfit./kap2 - ... 
                    reg./(ups.*kap2) - (n/2).*log(ups) - 2.*log(1 + ups);
        
   end
    
   rate = accept./maxiter;

end