% Train and test based on optical flow descriptors

parts = 3.5;

if (parts < 2)

    pkg load statistics;
    pkg load signal;
    pkg load image;

    n_vid = 1:180;%iii;%[1:1];
    Nfr = [-ones(1,25), ones(1,155)];%[-1, 1, 1];
    frames = 1:79;
    subs = 4;
    show_one_layer = 0;
    threshold = 10;

    clear opfx;
    clear opfy;
    if (show_one_layer)
        tt = time();
        [opfx, opfy, im1_pyr, im2_pyr] = multilayer_optical_flow_2_single(n_vid, frames, 10, subs, 3, 0);
        time() - tt
        figure;
        %imagesc(sqrt(sum(single(opfx2(:,:,:,1).^2+opfy2(:,:,:,1).^2),3)));
        imagesc(mean(single(opfx{4}(:,:,:,1)),3));
        figure;
        imagesc(mean(single(opfy{4}(:,:,:,1)),3));

        assert(false, 'OK !   OK !   OK !   OK !   OK !   OK !   OK !');
    end


    % Computes optical flow

    N = size(n_vid,2);
    frames_neg = frames(1,size(frames,2):-1:1);
    %optical flow descriptors;
    for n=1:N
        [opfx{1,n}, opfy{1,n}] = optical_flow_2_single(n_vid(1,n), frames, 20, subs, 5, 0);
        fprintf(2,'.');
        [opfx{2,n}, opfy{2,n}] = optical_flow_2_single(n_vid(1,n), frames_neg, 20, subs, 5, 0);
        fprintf(2,'.');
    end
    fprintf(2,'\n');
    
end



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

%C = 10; % SVM parameter
%mu = 15; % SVM parameter
%tol = 0.001; % Precision of SVM


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
                facto = 2;
                while (size(motions{f,n},2) > 25000)
                    keep = sum(abs(motions{f,n}),1) > threshold*facto;
                    fprintf("Proportion = %d\n", sum(keep,2)/size(keep,2));
                    motions{f,n} = motions{f,n}(:,keep);
                    facto = facto*2;
                end
                fprintf(2,'.');
                if (true)
                    a = motions{f,n};
                    fil = sprintf('batchs/motion_%d_%d.mat', n_vid(1,1), f);
                    %save(fil, 'a');
                end
            end
        end
        fprintf(2,'\n');
    end
end

if (parts == 3.5)
    clear motions;
    for n=1:180
        for f=1:2
            fil = sprintf('batchs/motion_%d_%d.mat', n, f);
            a = load(fil);
            motions{f, n} = a.a;
            fprintf(2,'.');
        end
    end
    fprintf(2,'\n');
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

    norhis = norm(histograms, 'columns');
    norhis = (norhis < 0.000001) + norhis;
    histograms = histograms ./ repmat(norhis,K,1);

    fr = reshape(repmat(Ffr',1,N) .* repmat(Nfr,F,1), 1, N*F);

end


% Now that we have histograms for each videos, we can do the train and test for different sets

assert(false);

load('a1.mat');
load('a2.mat');
N = 180;
F = 2;
C = 10; % SVM parameter
mu = 15; % SVM parameter
tol = 0.001; % Precision of SVM

    % Computes train and test sets
allC = [0.01, 0.1];%1, 10, 100, 1000, 10000, 100000];
n_ess = 5;

err_train = [0,0,err_train];%zeros(1,size(allC,2));
err_test = [0,0,err_test];%zeros(1,size(allC,2));

for ess2=1:size(allC,2)
    for ne=1:n_ess
        r = randperm(180);
        C = allC(1,ess2);
        for ess = 1:6
            ind = ones(180,1);%[1,1,1];
                              %r = randperm(180);
            r = circshift(r', 30)';
            ind = sum((r == (1:180)')(:,1:150),2)';
            
            ind = reshape(repmat(ind,F,1), 1, N*F);

            X = histograms(:,ind>0.5);
            y = fr(:,ind>0.5);

    % for i=1:size(ind,1)

            [w, wor] = svm_p(X, y, C, mu, tol);


            X = histograms;
            y = fr;

            fr_l = ((w'*X+wor>0)*2-1);



            err_train(1, ess2) += sum(fr_l.*fr.*ind < -0.5) / 300;
            err_test(1,ess2) += sum(fr_l.*fr.*(1-ind) < -0.5) / 60;
            fprintf(2, '.');
        end
        fprintf(2, ' ');
    end
    fprintf(2, '\n');
end

errtrain = err_train/6/n_ess   %0.33657
errtest = err_test/6/n_ess    %0.47267

allC = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000];

figure;
semilogx(allC, errtrain, '-k', 'linewidth', 3);
hold on;
semilogx(allC, errtest, '-r', 'linewidth', 3);
grid on;
xlabel('C parameter of the SVM');
ylabel('Error');
legend('Train', 'Test', 'location', 'southwest');
