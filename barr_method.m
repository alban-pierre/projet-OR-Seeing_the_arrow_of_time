function [x_sol, x_seq] = barr_method(Q,p,A,b,x_0,mu,tol)

    m = size(x_0,1);
    t = 0.1;
    stooop = 1;
    x_seq = x_0;
    
    while ((m/t>=tol/2) && (stooop < 1000))
	x_newseq = centering_step(Q, p, A, b, x_seq(:,end), t, tol/2);
	%size(x_newseq)
	x_seq = [x_seq, x_newseq];
	t = mu*t;
	stooop++;
    end
    assert(stooop < 1000);
    x_sol = x_seq(:,end);
end
