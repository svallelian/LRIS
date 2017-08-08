function xtGinvx = post_approx_q(x, L, V, l, mu, sigma)
% Evaluate the term x'*Gamma^{-1}*x
% Gamma approx (muA'A + sigma L'L)^{-1}
    
    Lx = L*x;
    lMat= sparse(diag(l));
    
    vec = sigma*Lx + mu*(V*lMat*(V'*Lx));
    xtGinvx = Lx'*vec;
    
end