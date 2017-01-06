function [opfx, opfy, im1_pyr, im2_pyr] = multilayer_optical_flow_2_single(n_vid, frames, alpha, subs, maxIter, blur)

    if (nargin < 6)
        blur = 0; % peut erte reduit à 3
    end
    if (nargin < 5)
        maxIter = 5; % peut erte reduit à 3
    end
    if (nargin < 4)
        subs = 1; % peut etre mis à 2 pour gagner un facteur ~3.
    end
    if (nargin < 4)
        alpha = 100;
    end
    
    if (n_vid < 10)
        dr = sprintf('ArrowDataAll/00%d/', n_vid);
    elseif (n_vid < 100)
        dr = sprintf('ArrowDataAll/0%d/', n_vid);
    else
        dr = sprintf('ArrowDataAll/%d/', n_vid);
    end

    
    f = frames(1,1);
    if (f < 10)
        filepathf = sprintf('%sim0000000%d.jpeg', dr, f);
    elseif (f < 100)
        filepathf = sprintf('%sim000000%d.jpeg', dr, f);
    else
        filepathf = sprintf('%sim00000%d.jpeg', dr, f);
    end
    im1 = imread(filepathf);
    im1 = single(im1);
    
    if (subs > 1)
        %s1 = (ceil(size(im1,1)/subs));
        %s2 = (ceil(size(im1,2)/subs));           
        %im(:,:,1) = downsample(downsample(conv2(im1(:,:,1),ones(subs)/(subs*subs))',subs,subs-1)',subs,subs-1);
        %im(:,:,2) = downsample(downsample(conv2(im1(:,:,2),ones(subs)/(subs*subs))',subs,subs-1)',subs,subs-1);
        %im(:,:,3) = downsample(downsample(conv2(im1(:,:,3),ones(subs)/(subs*subs))',subs,subs-1)',subs,subs-1);
        %if (size(im1,1) < s1*subs)
        %    im(end,:,:) = im(end,:,:) * subs/(size(im1,1) - (s1-1)*subs);
        %end
        %if (size(im1,2) < s2*subs)
        %    im(:,end,:) = im(:,end,:) * subs/(size(im1,2) - (s2-1)*subs);
        %end
        s1 = (floor(size(im1,1)/subs));
        s2 = (floor(size(im1,2)/subs));
        im = zeros(s1, s2, 3, 'single');
        im(:,:,1) = downsample(downsample(conv2(im1(:,:,1),ones(subs)/(subs*subs), 'valid')',subs)',subs);
        im(:,:,2) = downsample(downsample(conv2(im1(:,:,2),ones(subs)/(subs*subs), 'valid')',subs)',subs);
        im(:,:,3) = downsample(downsample(conv2(im1(:,:,3),ones(subs)/(subs*subs), 'valid')',subs)',subs);
        im1 = im;
    end
    
    im1 = im1/2;

    if (blur>0)
        [x, y] = meshgrid(-blur:blur,-blur:blur) ;
        g = exp((-x.^2 - y.^2)/blur);
        g = g ./ sum(sum(g));
        im1(:,:,1) = conv2(single(im1(:,:,1)),g,'same');
        im1(:,:,2) = conv2(single(im1(:,:,2)),g,'same');
        im1(:,:,3) = conv2(single(im1(:,:,3)),g,'same');
    end

    % compute pyramids
    clear im1_pyr;
    im1_pyr{1} = im1;
    sub = 2;
    for m=2:ceil(log2(min(size(im1(:,:,1))))-2)
        s1 = (ceil(size(im1_pyr{m-1},1)/sub));
        s2 = (ceil(size(im1_pyr{m-1},2)/sub));
        im = zeros(s1, s2, 3, 'single');
        im(:,:,1) = downsample(downsample(conv2(im1_pyr{m-1}(:,:,1),ones(sub)/(sub*sub))',sub,sub-1)',sub,sub-1);
        im(:,:,2) = downsample(downsample(conv2(im1_pyr{m-1}(:,:,2),ones(sub)/(sub*sub))',sub,sub-1)',sub,sub-1);
        im(:,:,3) = downsample(downsample(conv2(im1_pyr{m-1}(:,:,3),ones(sub)/(sub*sub))',sub,sub-1)',sub,sub-1);
        if (size(im1_pyr{m-1},1) < s1*sub)
            im(end,:,:) = im(end,:,:) * sub/(size(im1_pyr{m-1},1) - (s1-1)*sub);
        end
        if (size(im1_pyr{m-1},2) < s2*sub)
            im(:,end,:) = im(:,end,:) * sub/(size(im1_pyr{m-1},2) - (s2-1)*sub);
        end
        im1_pyr{m} = im;
        opfx{m} = zeros(size(im,1)-1, size(im,2)-1,3, size(frames,2)-1, 'single');
        opfy{m} = zeros(size(im,1)-1, size(im,2)-1,3, size(frames,2)-1, 'single');
    end
    
    %opfx = zeros(size(im1,1)-1, size(im1,2)-1,3, size(frames,2)-1, 'single');
    %opfy = zeros(size(im1,1)-1, size(im1,2)-1,3, size(frames,2)-1, 'single');
    opfx{1} = zeros(size(im1,1)-1, size(im1,2)-1,3, size(frames,2)-1, 'single');
    opfy{1} = zeros(size(im1,1)-1, size(im1,2)-1,3, size(frames,2)-1, 'single');
    iopf = 1;
    
    for f = frames(1,1:end-1)
        if (f+1 < 10)
            filepathf = sprintf('%sim0000000%d.jpeg', dr, f+1);
        elseif (f+1 < 100)
            filepathf = sprintf('%sim000000%d.jpeg', dr, f+1);
        else
            filepathf = sprintf('%sim00000%d.jpeg', dr, f+1);
        end
        im2 = imread(filepathf);
        im2 = single(im2);

        if (subs > 1)
            s1 = (floor(size(im2,1)/subs));
            s2 = (floor(size(im2,2)/subs));
            im = zeros(s1, s2, 3, 'single');
            im(:,:,1) = downsample(downsample(conv2(im2(:,:,1),ones(subs)/(subs*subs), 'valid')',subs)',subs);
            im(:,:,2) = downsample(downsample(conv2(im2(:,:,2),ones(subs)/(subs*subs), 'valid')',subs)',subs);
            im(:,:,3) = downsample(downsample(conv2(im2(:,:,3),ones(subs)/(subs*subs), 'valid')',subs)',subs);
            im2 = im;
        end
        im2 = im2/2;

        if (blur>0)
            im2(:,:,1) = conv2(single(im2(:,:,1)),g,'same');
            im2(:,:,2) = conv2(single(im2(:,:,2)),g,'same');
            im2(:,:,3) = conv2(single(im2(:,:,3)),g,'same');
        end

        clear im2_pyr;
        im2_pyr{1} = im2;
        sub = 2;
        for m=2:ceil(log2(min(size(im2(:,:,1))))-2)
            s1 = (ceil(size(im2_pyr{m-1},1)/sub));
            s2 = (ceil(size(im2_pyr{m-1},2)/sub));
            im = zeros(s1, s2, 3, 'single');
            im(:,:,1) = downsample(downsample(conv2(im2_pyr{m-1}(:,:,1),ones(sub)/(sub*sub))',sub,sub-1)',sub,sub-1);
            im(:,:,2) = downsample(downsample(conv2(im2_pyr{m-1}(:,:,2),ones(sub)/(sub*sub))',sub,sub-1)',sub,sub-1);
            im(:,:,3) = downsample(downsample(conv2(im2_pyr{m-1}(:,:,3),ones(sub)/(sub*sub))',sub,sub-1)',sub,sub-1);
            if (size(im2_pyr{m-1},1) < s1*sub)
                im(end,:,:) = im(end,:,:) * sub/(size(im2_pyr{m-1},1) - (s1-1)*sub);
            end
            if (size(im2_pyr{m-1},2) < s2*sub)
                im(:,end,:) = im(:,end,:) * sub/(size(im2_pyr{m-1},2) - (s2-1)*sub);
            end
            im2_pyr{m} = im;
        end
        
        % beginning of the multilayer interpolation
        for m=ceil(log2(min(size(im1_pyr{1}(:,:,1))))-2):-1:1 

            %sub = 2^m;
            %if (sub < 50)
            %    s1 = (ceil(size(im1_0,1)/sub));
            %    s2 = (ceil(size(im1_0,2)/sub));
            %   im = zeros(s1, s2, 3, 'single');
            %   im(:,:,1) = downsample(downsample(conv2(im1_0(:,:,1),ones(sub)/(sub*sub))',sub,sub-1)',sub,sub-1);
            %    im(:,:,2) = downsample(downsample(conv2(im1_0(:,:,2),ones(sub)/(sub*sub))',sub,sub-1)',sub,sub-1);
            %    im(:,:,3) = downsample(downsample(conv2(im1_0(:,:,3),ones(sub)/(sub*sub))',sub,sub-1)',sub,sub-1);
            %    if (size(im1_0,1) < s1*sub)
            %        im(end,:,:) = im(end,:,:) * sub/(size(im1_0,1) - (s1-1)*sub);
            %    end
            %    if (size(im1_,2) < s2*sub)
            %        im(:,end,:) = im(:,end,:) * sub/(size(im1_0,2) - (s2-1)*sub);
            %    end
            %else
            %    s1 = (ceil(size(im1_0,1)/sub));
            %    s2 = (ceil(size(im1_0,2)/sub));
            %   im = zeros(s1, s2, 3, 'single');
            %   im(:,:,1) = resample(resample(im1_0(:,:,1)',1,sub)',1,sub);
            %    im(:,:,2) = resample(resample(im1_0(:,:,2)',1,sub)',1,sub);
            %    im(:,:,3) = resample(resample(im1_0(:,:,3)',1,sub)',1,sub);
            %    assert(size(im,1) == s1);
            %    assert(size(im,2) == s2);
            %end
            %im1 = im;

            %if (sub < 50)
            %    s1 = (ceil(size(im2_0,1)/sub));
            %    s2 = (ceil(size(im2_0,2)/sub));
            %    im = zeros(s1, s2, 3, 'single');
            %   im(:,:,1) = downsample(downsample(conv2(im2_0(:,:,1),ones(sub)/(sub*sub))',sub,sub-1)',sub,sub-1);
            %    im(:,:,2) = downsample(downsample(conv2(im2_0(:,:,2),ones(sub)/(sub*sub))',sub,sub-1)',sub,sub-1);
            %    im(:,:,3) = downsample(downsample(conv2(im2_0(:,:,3),ones(sub)/(sub*sub))',sub,sub-1)',sub,sub-1);
            %    if (size(im2_0,1) < s1*sub)
            %        im(end,:,:) = im(end,:,:) * sub/(size(im2_0,1) - (s1-1)*sub);
            %    end
            %    if (size(im2_0,2) < s2*sub)
            %        im(:,end,:) = im(:,end,:) * sub/(size(im2_0,2) - (s2-1)*sub);
            %    end 
            %else
            %    s1 = (ceil(size(im2_0,1)/sub));
            %    s2 = (ceil(size(im2_0,2)/sub));
            %    im = zeros(s1, s2, 3, 'single');
            %   im(:,:,1) = resample(resample(im2_0(:,:,1)',1,sub)',1,sub);
            %    im(:,:,2) = resample(resample(im2_0(:,:,2)',1,sub)',1,sub);
            %    im(:,:,3) = resample(resample(im2_0(:,:,3)',1,sub)',1,sub);
            %    assert(size(im,1) == s1);
            %    assert(size(im,2) == s2);
            %end
            %im2 = im;

            im1 = im1_pyr{m};
            im2 = im2_pyr{m};
            
            fx = im2(2:end,2:end,:) + im2(1:end-1,2:end,:) - im2(2:end,1:end-1,:) - im2(1:end-1,1:end-1,:)  +  im1(2:end,2:end,:) + im1(1:end-1,2:end,:) - im1(2:end,1:end-1,:) - im1(1:end-1,1:end-1,:);
            fy = im2(2:end,2:end,:) - im2(1:end-1,2:end,:) + im2(2:end,1:end-1,:) - im2(1:end-1,1:end-1,:)  +  im1(2:end,2:end,:) - im1(1:end-1,2:end,:) + im1(2:end,1:end-1,:) - im1(1:end-1,1:end-1,:);
            ft = im2(2:end,2:end,:) + im2(1:end-1,2:end,:) + im2(2:end,1:end-1,:) + im2(1:end-1,1:end-1,:)  -  im1(2:end,2:end,:) - im1(1:end-1,2:end,:) - im1(2:end,1:end-1,:) - im1(1:end-1,1:end-1,:);
            
            if (m == ceil(log2(min(size(im1_pyr{1}(:,:,1))))-2))
                u = zeros(size(im1,1)-1, size(im1,2)-1,3, 'single');
                v = zeros(size(im1,1)-1, size(im1,2)-1,3, 'single');
            else
                u1 = zeros(size(im1,1)-1, size(im1,2)-1,3, 'single');
                v1 = zeros(size(im1,1)-1, size(im1,2)-1,3, 'single');
                s1 = 2*size(u,1);
                s2 = 2*size(u,2);
                u1(1:s1,1:s2,1) = resample(resample(u(:,:,1)',2,1)',2,1);
                u1(1:s1,1:s2,2) = resample(resample(u(:,:,2)',2,1)',2,1);
                u1(1:s1,1:s2,3) = resample(resample(u(:,:,3)',2,1)',2,1);
                u = u1(1:size(im1,1)-1, 1:size(im1,2)-1,:);
                v1(1:s1,1:s2,1) = resample(resample(v(:,:,1)',2,1)',2,1);
                v1(1:s1,1:s2,2) = resample(resample(v(:,:,2)',2,1)',2,1);
                v1(1:s1,1:s2,3) = resample(resample(v(:,:,3)',2,1)',2,1);
                v = v1(1:size(im1,1)-1, 1:size(im1,2)-1,:);
            end
            uav = zeros(size(im1,1)-1, size(im1,2)-1, 3, 'single');
            vav = zeros(size(im1,1)-1, size(im1,2)-1, 3, 'single');

            D = single(alpha^2 + fx.^2 + fy.^2);
            u = -(fx.*ft)./D;
            v = -(fy.*ft)./D;
            
            for it=1:maxIter
                uav = zeros(size(im1,1)-1, size(im1,2)-1, 3, 'single');
                uav(:,1:end-1,:) += (u(:,2:end,:))/4;
                uav(:,2:end,:) += (u(:,1:end-1,:))/4;
                uav(1:end-1,:,:) += (u(2:end,:,:))/4;
                uav(2:end,:,:) += (u(1:end-1,:,:))/4;
                vav = zeros(size(im1,1)-1, size(im1,2)-1, 3, 'single');
                vav(:,1:end-1,:) += (v(:,2:end,:))/4;
                vav(:,2:end,:) += (v(:,1:end-1,:))/4;
                vav(1:end-1,:,:) += (v(2:end,:,:))/4;
                vav(2:end,:,:) += (v(1:end-1,:,:))/4;

                P = single(fx.*uav + fy.*vav + ft);
                u = single(-(fx.*P)./D);
                v = single(-(fy.*P)./D);
            end

            opfx{m}(:,:,:,iopf) = u;
            opfy{m}(:,:,:,iopf) = v;
        
        end

        %opfx(:,:,:,iopf) = u;
        %opfy(:,:,:,iopf) = v;
        iopf++;

        im1 = im2;
    end
    
end
