function x_seq = centering_step(Q, p, A, b, x, t, tol)

    phi = @(x) (1/2*(x')*Q*x + p'*x);
    B = @(x) ([0,Inf](1,any(A*x - b >= 0)+1) + all(A*x - b < 0)*sum(-log(b-A*x),1));
    f = @(x,t) (t*phi(x) + B(x));

    grad = @(x,t) (t*(p + Q*x) + A'*(1./(b-A*x)));
    hess = @(x,t) (t*Q + A'*diag((1./(b-A*x)).^2)*A);
    
    gap = tol*10;
    stooop = 1;
    x_seq = x;
    
    while ((gap/2 >= tol) && (stooop < 1000))
	[x_new, gap] = newton(t,x_seq(:,end),f,grad,hess);
	x_seq = [x_seq, x_new];
	stooop++;
    end
    assert(stooop<1000);

end
