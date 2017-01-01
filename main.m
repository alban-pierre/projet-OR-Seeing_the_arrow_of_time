
[videos, Nfr, H, W, T] = load_dataset(1);
N = size(videos,2);

A = 2; % Number of angles in a descriptor
D = 4; % Size of a descriptor
K = 10; % Number of different descriptors after K-means
%N = 3; % Number of images to treat
%Nfr = [1,1,-1]; % Forward or reverse time in videos
F = 4; % Number of flips of each videos
Ffr = [1,1,-1,-1]; % Forward or reverse time in flips
%H = 100; % Video height
%W = 150; % Video width
%T = [10,6,9]; % Video time
Sh = 3; % Height subsampling 
Sw = 5; % Width subsampling 
St = 2; % Width subsampling 
Hs = ceil(H/Sh); % Video height after subsampling
Ws = ceil(W/Sw); % Video height after subsampling
Ts = ceil(T/St); % Video height after subsampling


    % Computation of motion descriptors of structure :
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


if (true)
	% Random initialisation
	clear motions;
	for n=1:N
		for f=1:F
			motions{f,n} = randn(D*D*A, Hs*Ws*Ts(1,n));
		end
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


[w, wor] = svm_p(X, y, C, mu, tol);

fr_l = ((w'*X+wor>0)*2-1);

