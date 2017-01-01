function [Q, p, A, b] = transform_svm_dual(C, X, y)

    n = size(X, 1);
    d = size(X, 2);

    Q = 1/2*(diag(y)*X*(X')*diag(y));
    p = -ones(n,1);
    A = [eye(n); -eye(n)];
    b = [ones(n,1)*C; zeros(n,1)];
    
end
