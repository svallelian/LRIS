function ratio = metropolisratio2(x, z, L, V_full, l_full, mu, k)
% Evaluates the Metropolis ratio using Proposition 2
% z is the proposed state
% x is the current state
% V_full, l_full are eigendecomposition of H
% k is eigenvalue cutoff
%
% D. Andrew Brown, Arvind Saibaba, Sarah Vallelian 07/2017

    Lz = L*z; Lx = L*x;
    evecs = V_full(:,k+1:end)'; 
    evals = l_full(k+1:end);
    
    vLz = evecs*Lz; 
    vLx = evecs*Lx;
    diff = vLz.^2 - vLx.^2;
    
    ratio = exp(-0.5*mu*(evals'*diff));
    
end