% Train and test based on rnn descriptors

parts = 1;

if (parts < 2)

    pkg load statistics;
    pkg load signal;
    pkg load image;

    addpath('DeepLearnToolbox-modified/CNN/');
    addpath('DeepLearnToolbox-modified/data/'); 
    addpath('DeepLearnToolbox-modified/util/');
    
    n_vid = [1:10,26:35];
    n_vid = [n_vid, n_vid];
    Nfr = [-ones(1,10), ones(1,10)];%[-1, 1, 1];
    Nfr = [Nfr, -Nfr];
    fli = [ones(1,20), -ones(1,20)];
    frames = 1:86;
    subs = 4;
    show_one_layer = 0;
    threshold = 10;
    W = 470;
    H = 262;

    N = size(n_vid,2)/2;
    T = size(frames, 2);
    B = 5;
    
    r = randperm(2*N);
    n_vid = n_vid(1,r);
    Nfr = Nfr(1,r);
    
    frames_neg = frames(1,size(frames,2):-1:1);

    rand('state',0)
    cnn.layers = {
                  struct('type', 'i') %input layer
                  struct('type', 'c', 'outputmaps', 6, 'kernelsize', 3) %convolution layer
                  struct('type', 's', 'scale', 2) %sub sampling layer
                  struct('type', 'c', 'outputmaps', 9, 'kernelsize', 3) %convolution layer
                  struct('type', 's', 'scale', 2) %subsampling layer
                  struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
                  struct('type', 's', 'scale', 2) %sub sampling layer
                  struct('type', 'c', 'outputmaps', 18, 'kernelsize', 7) %convolution layer
                  struct('type', 's', 'scale', 2) %subsampling layer
%                  struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
%                  struct('type', 's', 'scale', 2) %subsampling layer
%                  struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
%                  struct('type', 's', 'scale', 2) %subsampling layer
    };

    train_x = zeros(H, W, T, B);
    train_y = (Nfr(1,1:B) == [1;, -1]);
    cnn = cnnsetup(cnn, train_x, train_y);

    namesbatchs = ['batch1.mat'; 'batch2.mat'; 'batch3.mat'; 'batch4.mat'; 'batch5.mat'; 'batch6.mat'; 'batch7.mat'; 'batch8.mat'];
    
    opts.alpha = 1;
    opts.batchsize = B;
    opts.numepochs = 1;

    for nb=1:2*N/B
        train_x = zeros(H, W, T, B, 'single');
        for b=1:B
            fprintf(2,'.');
            n = (nb-1)*B + b;
            for f=frames
    %            train_x = zeros(H, W, T, n);
                if (fli(1, n) == 1)
                    train_x(:,:,f,b) = single(mean(single(loadimage(n_vid(1,n), f, H, W))/255,3));
                else
                    train_x(:,:,T-f+1,b) = single(mean(single(loadimage(n_vid(1,n), f, H, W))/255,3));
                end
            end
        end
        save(namesbatchs(nb,:), 'train_x');
        clear train_x;
    end
    fprintf(2,'\n');

    
    train_y = (Nfr == [1; -1]);
    %train_y = [train_y, 1-train_y];

    for nb=1:2*N/B
        load(namesbatchs(nb,:));
        cnn = cnntrain(cnn, train_x, train_y((nb-1)*B+1:nb*B,:), opts);
        clear train_x;
        fprintf(2,'.');
    end
   
    fprintf(2,'\n');

    size(cnn.fv)

    
    
end


assert(false);

    %[videos, Nfr, H, W, T] = load_dataset(1);
    %N = size(videos,2);

N = size(n_vid,2);

A = 2; % Number of angles in a descriptor
D = 4; % Size of a descriptor
K = 100; % Number of different descriptors after K-means
        %N = 3; % Number of images to treat
        %Nfr = [1,1,-1]; % Forward or reverse time in videos
F = 2; % Number of flips of each videos
Ffr = [1,-1];%,-1,-1]; % Forward or reverse time in flips
             %H = 100; % Video height
             %W = 150; % Video width
             %T = [10,6,9]; % Video time
Sh = 3; % Height subsampling 
Sw = 5; % Width subsampling 
St = 1; % Width subsampling 
        %Hs = floor((H-D+1)/Sh); % Video height after subsampling
        %Ws = floor((W-D+1)/Sw); % Video height after subsampling
        %Ts = floor((T-D+1)/St); % Video height after subsampling

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

if (parts < 3)

    if (false)
    % Random initialisation
        clear motions;
        for n=1:N
            for f=1:F
                motions{f,n} = randn(D*D*A, Hs(1,n)*Ws(1,n)*Ts(1,n));
            end
        end
    else
    % Computes motions descriptors
        clear motions;
        H = zeros(1,N);
        W = zeros(1,N);
        T = zeros(1,N);
        for n=1:N
            H(1,n) = size(opfx{1,n},1);
            W(1,n) = size(opfx{1,n},2);
            T(1,n) = size(opfx{1,n},4);
            Hs = floor((H-D+1)/Sh); % Video height after subsampling
            Ws = floor((W-D+1)/Sw); % Video height after subsampling
            Ts = floor(T/St); % Video height after subsampling
            for f=1:F
                i = 1;
                for dh=1:D
                    for dw=1:D
                        motions{f,n}(i,:) = reshape(mean(opfx{f,n},3)(dh+Sh*(0:Hs(1,n)-1), dw+Sw*(0:Ws(1,n)-1), 1, :), 1, Hs(1,n)*Ws(1,n)*Ts(1,n));
                        motions{f,n}(i+1,:) = reshape(mean(opfy{f,n},3)(dh+Sh*(0:Hs(1,n)-1), dw+Sw*(0:Ws(1,n)-1), 1, :), 1, Hs(1,n)*Ws(1,n)*Ts(1,n));
                        i = i+2;
                    end
                end
                keep = sum(abs(motions{f,n}),1) > threshold;
                fprintf("Proportion = %d\n", sum(keep,2)/size(keep,2));
                motions{f,n} = motions{f,n}(:,keep);
                fprintf(2,'.');
            end
        end
        fprintf(2,'\n');
        
    end
end

if (parts < 4)

    % Compute histograms
    [k, allk] = kmeans(cell2mat(reshape(motions, 1, N*F)), K, 0.1);

    %clear closestwords;
    histograms = zeros(K,F*N);
    for n=1:N
        for f=1:F
            dd = sqdist(k,motions{f,n});
            [~,imin] = min(dd,[],1);
            %closestwords{f,n} = reshape(imin, Hs(1,n)*Ws(1,n), Ts(1,n));
            histograms(:,(n-1)*F+f) = sum(repmat(imin,K,1)==repmat((1:K)', 1, size(imin,2)),2);%Hs(1,n)*Ws(1,n)*Ts(1,n)),2);
        end
    end

    histograms = histograms ./ repmat(norm(histograms, 'columns'),K,1);

    fr = reshape(repmat(Ffr',1,N) .* repmat(Nfr,F,1), 1, N*F);

end


% Now that we have histograms for each videos, we can do the train and test for different sets


% Computes train and test sets
ind = ones(20,1);%[1,1,1];
r = randperm(20);
ind = sum((r == (1:20)')(:,1:15),2)';

ind = reshape(repmat(ind,F,1), 1, N*F);

X = histograms(:,ind>0.5);
y = fr(:,ind>0.5);

    % for i=1:size(ind,1)

[w, wor] = svm_p(X, y, C, mu, tol);


X = histograms;
y = fr;

fr_l = ((w'*X+wor>0)*2-1)



err_train = sum(fr_l.*fr.*ind < -0.5) / 30
err_test = sum(fr_l.*fr.*(1-ind) < -0.5) / 10
