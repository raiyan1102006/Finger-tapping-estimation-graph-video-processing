clc; clear all; close all;

%% Loading data

colorPen = VideoReader('data/test2.mov');
get(colorPen);

StartFrame=1;
EndFrame=get(colorPen, 'NumberOfFrames');
rescale_height = 480; %resizing each frame to this height, keeping aspect ratio intact

%% Displaying a sample frame

rgbFrame0 = read(colorPen, round(EndFrame/2));
rgbFrame0  = imresize(rgbFrame0 , [rescale_height NaN]);
rgbFrame0 = rgb2gray(rgbFrame0);

subplot(221)
imshow(rgbFrame0);
title('(a) Sample Frame');

%% Creating a 4D tensor that holds the pre-processed video data

vid_tensor = zeros(size(rgbFrame0,1),size(rgbFrame0,2),EndFrame);
for i=StartFrame:EndFrame
    rgbFrame0 = read(colorPen, i);
    rgbFrame0  = imresize(rgbFrame0 , [rescale_height NaN]);
    rgbFrame0 = rgb2gray(rgbFrame0);
    vid_tensor(:,:,i) =  rgbFrame0;
end

%% Calculating total variation for each pixel location

TV = zeros(size(rgbFrame0,1),size(rgbFrame0,2));
for i=StartFrame+1:EndFrame-1
    for row=1:size(rgbFrame0,1)
        for col=1:size(rgbFrame0,2)
            TV(row,col) = TV(row,col)+(vid_tensor(row,col,i)-vid_tensor(row,col,i-1))^2+(vid_tensor(row,col,i)-vid_tensor(row,col,i+1))^2;
        end
    end
end

subplot(222)
mesh(flipud(TV));
zlabel('Total Variation')
title('(b) Mesh plot of Total Variation of each pixel location');

% Grayscale approximation of Total Variation
TV2 = uint8(TV.*( (255)/(max(max(TV))) ));
subplot(223);
imshow(TV2)
title('(c) Grayscale approximation of Total Variation');

% Binary approximation of Total Variation
TV_reshaped = reshape(TV,1,[]);
Y = prctile(TV_reshaped,95);
TV_ROI = TV>Y;
subplot(224)
imshow(TV_ROI)
title('(d) Binary Approximation of Total Variation')

%% Image segmentation

% extracting information of two types: boundingbox and image
Iprops=regionprops(TV_ROI,'BoundingBox','Image'); 
NR=cat(1,Iprops.BoundingBox); 

% finding the bounding box with maximum area
NR2 = [NR NR(:,3).*NR(:,4)]; 
index_max_bbox = find(NR2(:,5)==max(NR2(:,5)));

% keeping a buffer region around the selected bounding boxes
buffer_perc = 1;

x_buffer = size(rgbFrame0,2)*buffer_perc/100;
y_buffer = size(rgbFrame0,1)*buffer_perc/100;
up_left_x=NR2(index_max_bbox,1)-x_buffer;
up_left_y=NR2(index_max_bbox,2)-y_buffer;
width=NR2(index_max_bbox,3)+2*x_buffer;
height=NR2(index_max_bbox,4)+2*y_buffer;
rng default;

% Cutting out the selected bounding boxes

TT_dummy = imcrop(rgbFrame0,[up_left_x up_left_y width height]); 
TT_tensor = zeros(size(TT_dummy,1),size(TT_dummy,2),EndFrame);

for frame=StartFrame:EndFrame
    rgbFrame0 = read(colorPen, frame);
    rgbFrame0  = imresize(rgbFrame0 , [rescale_height NaN]);
    TT=imcrop(rgbFrame0,[up_left_x up_left_y width height]); 
    TT = rgb2gray(TT);
    TT_tensor(:,:,frame) = TT;
end

%% Processing the cropped image segments

TT_tensor_diff = zeros(size(TT_dummy,1),size(TT_dummy,2),EndFrame-1);
for frame=StartFrame:EndFrame-1
    TT_tensor_diff(:,:,frame) = TT_tensor(:,:,frame)-TT_tensor(:,:,frame+1);
end

TV_fft = zeros(size(TT_tensor_diff,1),size(TT_tensor_diff,2));
for i=StartFrame+1:EndFrame-1
    for row=1:size(TT_tensor_diff,1)
        for col=1:size(TT_tensor_diff,2)
            TV_fft(row,col) = TV_fft(row,col)+(TT_tensor(row,col,i)-TT_tensor(row,col,i-1))^2+(TT_tensor(row,col,i)-TT_tensor(row,col,i+1))^2;
        end
    end
end

TV_fft2 = uint8(TV_fft.*( (255)/(max(max(TV_fft))) ));
figure;imshow(TV_fft2)
title('Grayscale heatmap of TV in the cropped image')

% Binarization of ROI
TV_fft_reshaped = reshape(TV_fft,1,[]);
g1=85;
g2=100;
TV_fft_Y_1 = prctile(TV_fft_reshaped,g1);
TV_fft_Y_2 = prctile(TV_fft_reshaped,g2);
TV_fft_ROI1 = TV_fft<TV_fft_Y_1;
TV_fft_ROI2 = TV_fft>TV_fft_Y_2;
TV_fft_ROI = TV_fft_ROI1+TV_fft_ROI2;
TV_fft_ROI = TV_fft_ROI==0;
figure;imshow(TV_fft_ROI)
title('Binary approximation at 85th percentile of TV')

%% Extracting finger-tapping frequency using FFT and GFT

Fs = colorPen.FrameRate; %sampling frequency
T = 1/Fs; %sampling period
L = EndFrame; %length of signal
t = (0:L-1)*T; %time vector

fft_arr = [];
comb=[];
L1=EndFrame-1;

% Creating adjacency matrix
A=zeros(L1,L1);
A(1,2)=1;
A(L1,L1-1)=1;

for i=2:L1-1
    A(i,i-1)=1;   
    A(i,i+1)=1;   
end

La= diag(A*ones(L1,1))-A; % Laplacian Matrix
[V,D] = eig(La); % Eigenvectors of the Laplacian
f2 = Fs*(1:L1)/(2*L1);

for row=1:size(TT_tensor_diff,1)
    for col=1:size(TT_tensor_diff,2)
        if(TV_fft_ROI(row,col)==1)
            A = TT_tensor_diff(row,col,:);
            A = A(:);
            comb=[comb A];
            
            %Using FFT
            Y = fft(A);
            P2 = abs(Y/L);
            P1 = P2(1:L/2+1);
            P1(2:end-1) = 2*P1(2:end-1);
            f = Fs*(0:(L/2))/L;
            [pks,locs] = findpeaks(P1,'NPeaks',1,'SortStr','descend');
            fft_arr=[fft_arr f(locs)];
            
            % Uncomment the following three lines to run GFT instead of
            % FFT. Comment out the FFT segment above.
           
            %f_hat = V'*A;
            %[pks,locs] = findpeaks(f_hat,'NPeaks',1,'SortStr','descend');
            %fft_arr=[fft_arr f2(locs)];
            
        end
    end
end

% Using a 'voting' scheme to determine the final frequency
finger_tapping_frequency = mode(fft_arr)