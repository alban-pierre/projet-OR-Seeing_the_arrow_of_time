

A = 2; % Number of angles in a descriptor
D = 4; % Size of a descriptor
K = 10; % Number of different descriptors after K-means
N = 3; % Number of images to treat
Nfr = [1,1,-1]; % Forward or reverse time in videos
F = 4; % Number of flips of each videos
Ffr = [1,1,-1,-1]; % Forward or reverse time in flips
H = 100; % Video height
W = 150; % Video width
T = [10,6,9]; % Video time
Sh = 3; % Height subsampling 
Sw = 5; % Width subsampling 
St = 2; % Width subsampling 
Hs = ceil(H/Sh); % Video height after subsampling
Ws = ceil(W/Sw); % Video height after subsampling
Ts = ceil(T/St); % Video height after subsampling


    % TODO Add here the computation of motion descriptors of structure :
    % {n1f1, n2f1, n3f1, ..., nNf1;
    %  n1f2, n2f2, n3f2, ..., nNf2;
    %  n1f3, n2f3, n3f3, ..., nNf3;
    %  n1f4, n2f4, n3f4, ..., nNf4}
    %         |
    %  ______/ \___________
    % /                    \
    % [t1, t2, t3, ..., tTs]
    %      |
    %  ___/ \______________
    % /                    \
    % [w1, w2, w3, ..., wWs]
    %      |
    %  ___/ \______________
    % /                    \
    % [h1, h2, h3, ..., hHs]

clear motions;
for n=1:N
    for f=1:F
        motions{f,n} = randn(D*D*A, Hs*Ws*Ts(1,n));
    end
end

[k, allk] = kmeans(cell2mat(reshape(motions, 1, n*f)), K, 0.1);

clear closestwords;
histograms = zeros(K,F*N);
for n=1:N
    for f=1:F
        dd = sqdist(k,motions{f,n});
        [~,imin] = min(dd,[],1);
        closestwords{f,n} = reshape(imin, Hs*Ws, Ts(1,n));
        histograms(:,(n-1)*F+f) = sum(repmat(imin,K,1)==repmat((1:10)', 1, Hs*Ws*Ts(1,n)),2);
    end
end

histograms = histograms ./ repmat(norm(histograms, 'columns'),K,1);

fr = reshape(repmat(Ffr',1,N) .* repmat(Nfr,F,1), 1, N*F);

X = histograms;
y = fr;
C = 10;
mu = 15;
tol = 0.001;


X = [randn(2,300), [2;2]+randn(2,300)];
y = [ones(1,300), -ones(1,300)];
tt = time();
[w, wor] = svm_p(X, y, C, mu, tol);
time() - tt
tt = time();
[w2, wor2] = svm_d(X, y, C, mu, tol);
time() - tt

figure;
plot(X(1,w'*X+wor>0), X(2,w'*X+wor>0), '.b')
hold on;
plot(X(1,w'*X+wor<0), X(2,w'*X+wor<0), '.r')

figure;
plot(X(1,w2'*X+wor2>0), X(2,w2'*X+wor2>0), '.b')
hold on;
plot(X(1,w2'*X+wor2<0), X(2,w2'*X+wor2<0), '.r')




assert(false);




n = size(X,1);
d = size(X,2);

[Q,p,A,b] = transform_svm_primal(C,X,y);
x_0 = [zeros(d,1); 2*ones(n,1)+3*rand(n,1)];
assert(all(A*x_0 - b <= 0));
[x_sol, x_seq] = barr_method(Q,p,A,b,x_0,mu,tol);
wp = x_sol(1:3,1);
phi = @(x) (1/2*(x')*Q*x + p'*x);
clear phixp;
itp = size(x_seq,2);
for i=1:itp
    phixp(1,i) = phi(x_seq(:,i));
end

assert(false);


[Q,p,A,b] = transform_svm_dual(C,X,y);
x_0 = [C/4 + C/2*rand(n,1)];
assert(all(A*x_0 - b <= 0));
[x_sol, x_seq] = barr_method(Q,p,A,b,x_0,mu,tol);
wd = X'*(x_sol.*y);
w_seqd = X'*(x_seq.*repmat(y,1,size(x_seq,2)));
z_seqd = max(1-repmat(y,1,size(x_seq,2)).*(X*w_seqd),0);
clear phixd;
itd = size(x_seq,2);
for i=1:itd
    phixd(1,i) = 1/2*norm(w_seqd(:,i)) + C*sum(z_seqd(:,i),1);
end

