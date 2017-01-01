function [x_new, gap] = newton(t,x,f,grad,hess)

    g = grad(x,t);
    h = hess(x,t);
    fx = f(x,t);

    dxnt = -(h^-1)*g;

    %alpha entre 0 et 1/2
    %beta entre 0 et 1
    alpha = 1/4;
    beta = 1/2;

    bt = 1;
    stooop = 1;
    while ((f(x+bt*dxnt,t) >= fx + alpha*bt*g'*dxnt) && (stooop < 1000))
	bt *= beta;
	stooop++;
    end
    assert(stooop<1000);

    if (stooop <= 1)
	stooop = 1;
	while ((f(x+(bt/beta)*dxnt,t) < f(x+bt*dxnt,t)) && (stooop < 1000))
	    bt /= beta;
	    stooop++;
	end
	assert(stooop<1000);
    end
    x_new = x+bt*dxnt;
    gap = -bt*g'*dxnt;
end
