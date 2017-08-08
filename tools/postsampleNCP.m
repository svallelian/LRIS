function y = postsampleNCP(Atb,L,V,l,mu,sigma,epsilon)
% Sample approximately from a Gaussian distribution N(mean,Gamma)
% Gamma = (mu*A'A+sigma*L'*L)
% mean  = Gamma\(mu*A'b)
%
% Modified to sample z with the non-centered parameterization under Gamma
% priors; see Supplementary Materials 2.1 for details.
    
    d = mu*l./(sigma+l*mu);
    dg = 1 - sqrt(1-d);
    
    n = length(dg);
    ddg = spdiags(dg,0,n,n);
    
    % NCP
    mean = postinv((mu/sqrt(sigma))*(Atb),L,V,l,mu/sigma,1);
    dev = L\(epsilon - V*ddg*(V'*epsilon))/sqrt(1);
    
    y = mean + dev;
    
end
