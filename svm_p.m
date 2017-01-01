function [w, wor] = svm_p(X, y, C, mu, tol)

    % Computes svm according to the primal

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
    
    [Q,p,A,b] = transform_svm_primal(C,X',y');
    x_0 = [zeros(D+1,1); 2*ones(N,1)+3*rand(N,1)];
    assert(all(A*x_0 - b <= 0));
    [x_sol, x_seq] = barr_method(Q,p,A,b,x_0,mu,tol);
    w = x_sol(1:D,1);
    wor = x_sol(D+1,1);
end
