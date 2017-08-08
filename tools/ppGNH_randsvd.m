function [V,l] = ppGNH_randsvd(A,L,k,p)
% Uses randomized SVD to compute the eigenpairs of the matrix L^{-T}A'AL^{-1}
% Corresponding to the k largest eigenvalues 
% With oversampling factor p

    [~,s,v] = randsvd(A, L, k, p);
    V = v(:,1:k);
    l = s(1:k).^2;
end

function [u,s,v] = randsvd(A,L,k,p)
    [m,n] = size(A);
    ell = k + p;
    
    Omega = randn(n,ell);
    
    Y = zeros(m,ell);
    for i = 1:k
        Y(:,i) = A*(L\(Omega(:,i)));
    end
    
    [q,~] = qr(Y, 0);
    
    B = zeros(n,ell);
    for i = 1:ell
        B(:,i) = L'\(A'*q(:,i));
    end
    
    [u,s,v] = svd(B','econ');
    u = q*u;
    s = diag(s);

end