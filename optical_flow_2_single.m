function [opfx, opfy] = optical_flow_2_single(n_vid, frames, alpha, subs, maxIter)

    if (nargin < 5)
        maxIter = 5; % peut erte reduit � 3
    end
    if (nargin < 4)
        subs = 1; % peut etre mis � 2 pour gagner un facteur ~3.
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
        s1 = (floor(size(im1,1)/subs));
        s2 = (floor(size(im1,2)/subs));           
        im1 = im1(1:subs*s1, 1:subs*s2, :);
        im = zeros(s1, s2, 3, 'single');
        im(:,:,1) = reshape(mean(reshape(reshape(mean(reshape(im1(:,:,1),subs,s1*subs*s2),1),s1,s2*subs)',subs,s1*s2),1),s2,s1)';
        im(:,:,2) = reshape(mean(reshape(reshape(mean(reshape(im1(:,:,2),subs,s1*subs*s2),1),s1,s2*subs)',subs,s1*s2),1),s2,s1)';
        im(:,:,3) = reshape(mean(reshape(reshape(mean(reshape(im1(:,:,3),subs,s1*subs*s2),1),s1,s2*subs)',subs,s1*s2),1),s2,s1)';
        im1 = im;
    end
    
    im1 = im1/2;
    
    opfx = zeros(size(im1,1)-1, size(im1,2)-1,3, size(frames,2)-1, 'single');
    opfy = zeros(size(im1,1)-1, size(im1,2)-1,3, size(frames,2)-1, 'single');
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
            im2 = im2(1:subs*s1, 1:subs*s2, :);
            im = zeros(s1, s2, 3, 'single');
            im(:,:,1) = reshape(mean(reshape(reshape(mean(reshape(im2(:,:,1),subs,s1*subs*s2),1),s1,s2*subs)',subs,s1*s2),1),s2,s1)';
            im(:,:,2) = reshape(mean(reshape(reshape(mean(reshape(im2(:,:,2),subs,s1*subs*s2),1),s1,s2*subs)',subs,s1*s2),1),s2,s1)';
            im(:,:,3) = reshape(mean(reshape(reshape(mean(reshape(im2(:,:,3),subs,s1*subs*s2),1),s1,s2*subs)',subs,s1*s2),1),s2,s1)';
            im2 = im;
        end
        im2 = im2/2;
        
        %fx = zeros(size(im1,1)-2, size(im1,2)-2,3, 'int8');
        %fy = zeros(size(im1,1)-2, size(im1,2)-2,3, 'int8');
        %ft = zeros(size(im1,1)-2, size(im1,2)-2,3, 'int8');

        fx = im2(2:end,2:end,:) + im2(1:end-1,2:end,:) - im2(2:end,1:end-1,:) - im2(1:end-1,1:end-1,:)  +  im1(2:end,2:end,:) + im1(1:end-1,2:end,:) - im1(2:end,1:end-1,:) - im1(1:end-1,1:end-1,:);
        fy = im2(2:end,2:end,:) - im2(1:end-1,2:end,:) + im2(2:end,1:end-1,:) - im2(1:end-1,1:end-1,:)  +  im1(2:end,2:end,:) - im1(1:end-1,2:end,:) + im1(2:end,1:end-1,:) - im1(1:end-1,1:end-1,:);
        ft = im2(2:end,2:end,:) + im2(1:end-1,2:end,:) + im2(2:end,1:end-1,:) + im2(1:end-1,1:end-1,:)  -  im1(2:end,2:end,:) - im1(1:end-1,2:end,:) - im1(2:end,1:end-1,:) - im1(1:end-1,1:end-1,:);
        
        u = zeros(size(im1,1)-1, size(im1,2)-1,3, 'single');
        v = zeros(size(im1,1)-1, size(im1,2)-1,3, 'single');
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

        opfx(:,:,:,iopf) = u;
        opfy(:,:,:,iopf) = v;
        iopf++;

        im1 = im2;
    end
    
end
