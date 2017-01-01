function [Q, p, A, b] = transform_svm_primal(C, X, y)

    n = size(X, 1);
    d = size(X, 2);

    Q = diag([ones(1,d)/2, zeros(1,n)]);
    p = [zeros(d,1); ones(n,1)*C];
    A = [-diag(y)*X, -eye(n); zeros(n,d), -eye(n)];
    b = [-ones(n,1); zeros(n,1)];
    
end
