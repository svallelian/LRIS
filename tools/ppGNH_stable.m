function [V,l] = ppGNH_stable(A, L, k)
% Computes the eigenpairs of the matrix L^{-T}A'AL^{-1}
% Corresponding to the k largest eigenvalues 
 
    [m,n] = size(A);
    
    % Use the cyclic matrix [0 A; A' 0] approach
    opts.isreal = 1;
    opts.issym = 1;
    [Vr,D] = eigs(@(x)B(x,A,L),m+n,2*k,'LM',opts); 
    
    % Return desired eigenvalues and eigenvectors
    l = diag(D);    l = l(1:k).^2;
    V = Vr(m+1:end,1:k)*sqrt(2);
    
    
end

function z = B(x,A,L)
    [m,n] = size(A);
    z = zeros(m+n,1);
    
    z(1:m) = A*(L\x(m+1:end));
    z(m+1:end) = L'\(A'*x(1:m));
    
end