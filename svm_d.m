function [w, wor] = svm_d(X, y, C, mu, tol)

    % Computes svm according to the dual

    % Dimensions :
    % N : Number of points
    % D : Dimension of points

    % Input :
    % X   : (D*N)  : Points
    % y   : (1*N)  : Cluster 
    % C   : Double : Constant of SVM
    % mu  : Double : Division of barrier between each minimization
    % tol : Double : Precision of the output

    % Output :
    % w   : (D*1)  : Coefficient of classification
    % wor : Double : Offset for classification

    [D, N] = size(X);

    X = [X; ones(1,N)];
    
    [Q,p,A,b] = transform_svm_dual(C,X',y');
    x_0 = [C/4 + C/2*rand(N,1)];
    assert(all(A*x_0 - b <= 0));
    [x_sol, x_seq] = barr_method(Q,p,A,b,x_0,mu,tol);
    wd = X*(x_sol.*y');
    w = wd(1:end-1,1);
    wor = wd(end,1);
end
