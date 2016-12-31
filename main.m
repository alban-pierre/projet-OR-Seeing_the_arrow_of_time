

A = 2; % Number of angles in a descriptor
D = 4; % Size of a descriptor
K = 10; % Number of different descriptors after K-means
N = 1; % Number of images to treat
W = 100; % Video width
H = 100; % Video height
T = 10; % Video time
Sw = 2; % Width subsampling 
Sh = 2; % Height subsampling 
St = 2; % Width subsampling 


motions = randn(D*D*A, ceil(H/Sh)*ceil(W/Sw)*ceil(T/St));




