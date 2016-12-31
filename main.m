

A = 2; % Number of angles in a descriptor
D = 4; % Size of a descriptor
K = 10; % Number of different descriptors after K-means
N = 3; % Number of images to treat
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
    % {n1, n2, n3, ..., nN}
    %      |
    %  ___/ \______________
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
    motions{n} = randn(D*D*A, Hs*Ws*Ts(1,n));
end

[k, allk] = kmeans(cell2mat(motions),K,0.1);

clear histograms;
for n=1:N
	dd = sqdist(k,motions{n});
	[~,imin] = min(dd,[],1);
    histograms{n} = reshape(imin, Hs*Ws, Ts(1,n));
end

% If it is faster
%dd = sqdist(k,cell2mat(motions));
%[~,imin] = min(dd,[],1);
%histograms = reshape(imin, Hs*Ws, sum(Ts,2));
