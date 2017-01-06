
n_vid = 1;
frames = 1:3;
subs = 1;

for m=[]
    [opfx2, opfy2] = optical_flow_3_single(1, 1:3, 100, 1, m);
    figure;
    imagesc(sqrt(sum(single(opfx2(:,:,:,1).^2+opfy2(:,:,:,1).^2),3)));
end

%tt = time();
%[opfx1, opfy1] = optical_flow_3_int8(n_vid, frames, 3, subs, 3);
%time() - tt
%figure;
%imagesc(sqrt(sum(single(opfx1(:,:,:,1).^2+opfy1(:,:,:,1).^2),3)));

tt = time();
[opfx2, opfy2] = optical_flow_3_single(n_vid, frames, 100, subs, 5);
time() - tt
figure;
%imagesc(sqrt(sum(single(opfx2(:,:,:,1).^2+opfy2(:,:,:,1).^2),3)));
imagesc(sum(single(opfx2(:,:,:,1)),3));
figure;
imagesc(sum(single(opfy2(:,:,:,1)),3));

assert(false);

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

C = 10; % SVM parameter
mu = 15; % SVM parameter
tol = 0.001; % Precision of SVM


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
            motions{f,n} = randn(D*D*A, Hs(1,n)*Ws(1,n)*Ts(1,n));
        end
    end
end




% Compute histograms
[k, allk] = kmeans(cell2mat(reshape(motions, 1, n*f)), K, 0.1);

clear closestwords;
histograms = zeros(K,F*N);
for n=1:N
    for f=1:F
        dd = sqdist(k,motions{f,n});
        [~,imin] = min(dd,[],1);
        closestwords{f,n} = reshape(imin, Hs(1,n)*Ws(1,n), Ts(1,n));
        histograms(:,(n-1)*F+f) = sum(repmat(imin,K,1)==repmat((1:10)', 1, Hs(1,n)*Ws(1,n)*Ts(1,n)),2);
    end
end

histograms = histograms ./ repmat(norm(histograms, 'columns'),K,1);

fr = reshape(repmat(Ffr',1,N) .* repmat(Nfr,F,1), 1, N*F);




% Now that we have histograms for each videos, we can do the train and test for different sets


% Computes train and test sets
ind = [1,0,1];


X = histograms;
y = fr;

% for i=1:size(ind,1)

[w, wor] = svm_p(X, y, C, mu, tol);

fr_l = ((w'*X+wor>0)*2-1);

