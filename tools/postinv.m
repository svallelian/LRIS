function y = postinv(x,L,V,l,mu,sigma)
% Apply approximate posterior covariance to the vector x
% Gamma = (muA'A + sigma L'L)^{-1}
    
    d = mu*l./(sigma+l.*mu);
    n = length(d);
    D = spdiags(d,0,n,n);
    
    Ltx = L'\x;
    y = L\(Ltx - V*D*(V'*Ltx));
    y = y./sigma;

end