function ratio = metropolisratio(x, z, A, L, V, l, mu, sigma)
% Evaluates the Metropolis ratio w(z)/w(x)
% z is the proposed state
% x is the current state
%
% D. Andrew Brown, Arvind Saibaba, Sarah Vallelian 07/2017
    
    %Compute log w(z)
    Az = A*z;   Lz = L*z;
    lwz1 = mu*(Az'*Az) + sigma*(Lz'*Lz); 
    lwz2 = post_approx_q(z, L, V, l, mu, sigma);
    lwz  = -0.5*(lwz1 - lwz2);
    
    %Compute log w(x)
    Ax = A*x;   Lx = L*x;
    lwx1 = mu*(Ax'*Ax) + sigma*(Lx'*Lx); 
    lwx2 = post_approx_q(x, L, V, l, mu, sigma);
    lwx  = -0.5*(lwx1 -lwx2);
    
    ratio = exp(lwz - lwx);
    
end