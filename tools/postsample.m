function y = postsample(Atb,L,V,l,mu,sigma,epsilon)
% Sample approximately from a Gaussian distribution N(mean,Gamma)
% Gamma = (mu*A'A+sigma*L'*L)
% mean  = Gamma\(mu*A'b)
    
    d = mu*l./(sigma+l*mu);
    dg = 1 - sqrt(1-d);
    
    n = length(dg);
    ddg = spdiags(dg,0,n,n);
    
    mean = postinv(mu*(Atb),L,V,l,mu,sigma);
    dev = L\(epsilon - V*ddg*(V'*epsilon))/sqrt(sigma);
    
    y = mean + dev;
    
end
