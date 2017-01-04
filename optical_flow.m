function opf = optical_flow(n_vid, frames, alpha, subsampling, maxIter)

    if (nargin < 5)
        maxIter = 10;
    end
    if (nargin < 4)
        subsampling = 1;
    end
    if (nargin < 4)
        alpha = 3;
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
    
    f = frames(1,2);
    if (f < 10)
        filepathf = sprintf('%sim0000000%d.jpeg', dr, f);
    elseif (f < 100)
        filepathf = sprintf('%sim000000%d.jpeg', dr, f);
    else
        filepathf = sprintf('%sim00000%d.jpeg', dr, f);
    end
    im2 = imread(filepathf);
    
    im1 = im1/2;
    im2 = im2/2;
    
    opf = zeros(size(im1,1)-2, size(im1,2)-2,3, size(frames,2), 'int16');
    iopf = 1;
    
    for f = frames(1,2:end-1)
        if (f+1 < 10)
            filepathf = sprintf('%sim0000000%d.jpeg', dr, f+1);
        elseif (f+1 < 100)
            filepathf = sprintf('%sim000000%d.jpeg', dr, f+1);
        else
            filepathf = sprintf('%sim00000%d.jpeg', dr, f+1);
        end
        im3 = imread(filepathf);
        im3 = im3/2;
        
        %fx = zeros(size(im1,1)-2, size(im1,2)-2,3, 'int8');
        %fy = zeros(size(im1,1)-2, size(im1,2)-2,3, 'int8');
        %ft = zeros(size(im1,1)-2, size(im1,2)-2,3, 'int8');
        fx = int8(im3(2:end-1,3:end,:)) - int8(im1(2:end-1,1:end-2,:));
        fy = int8(im3(3:end,2:end-1,:)) - int8(im1(1:end-2,2:end-1,:));
        ft = int8(im3(2:end-1,2:end-1,:)) - int8(im1(2:end-1,2:end-1,:));
        
        u = zeros(size(im1,1)-2, size(im1,2)-2,3, 'int8');
        v = zeros(size(im1,1)-2, size(im1,2)-2,3, 'int8');
        uav = zeros(size(im1,1)-2, size(im1,2)-2, 3, 'int8');
        vav = zeros(size(im1,1)-2, size(im1,2)-2, 3, 'int8');

        D = int16(alpha^2 + fx.^2 + fy.^2);
        u = int8(-int16(fx.*ft)./D);
        v = int8(-int16(fy.*ft)./D);
        
        for it=1:maxIter
            uav = zeros(size(im1,1)-2, size(im1,2)-2, 3, 'int8');
            uav(:,1:end-1,:) += (u(:,2:end,:))/4;
            uav(:,2:end,:) += (u(:,1:end-1,:))/4;
            uav(1:end-1,:,:) += (u(2:end,:,:))/4;
            uav(2:end,:,:) += (u(1:end-1,:,:))/4;
            vav = zeros(size(im1,1)-2, size(im1,2)-2, 3, 'int8');
            vav(:,1:end-1,:) += (v(:,2:end,:))/4;
            vav(:,2:end,:) += (v(:,1:end-1,:))/4;
            vav(1:end-1,:,:) += (v(2:end,:,:))/4;
            vav(2:end,:,:) += (v(1:end-1,:,:))/4;

            P = int8(fx.*uav + fy.*vav + ft);
            u = int8(-int16(fx.*P)./D);
            v = int8(-int16(fy.*P)./D);
        end

        opf(:,:,:,iopf) = int16(u.^2 + v.^2);
        iopf++;

        im1 = im2;
        im2 = im3;
    end
    
end
