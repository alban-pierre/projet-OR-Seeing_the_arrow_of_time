function [videos, Nfr, H, W, T] = load_dataset(n)

    % Load a specific dataset

    % Dimensions :
    % N : Number of videos
    % Hn : Height of the nth video
    % Wn : Width of the nth video
    % Tn : Time lenght of the nth video
    
    % Input :
    % n : Int : Identification number of the dataset

    % Output :
    % videos : {1*N}(Hn*Wn*Tn) : Videos in the dataset
    % Nfr    : (1*N)           : Forward or reverse time of videos
    % H      : (1*N)           : Height of videos
    % W      : (1*N)           : Width of videos
    % T      : (1*N)           : Time of videos

    if (n == 1)
        H = [100,90,90];
        W = [150,150,120];
        T = [10,6,9];
        Nfr = [1,1,-1];
        for n=1:3
            videos{1,n} = zeros(H(1,n)*W(1,n), T(1,n));
        end
    end

end
