function [] = showEmbedding(points, imgNames, imgDatasets, boxes, correctInds)

globals;
if(~exist('correctInds','var'))
    correctInds = ones(size(imgNames));
    visColormap = [1 1 1];
else
    visColormap = [1 1 1; 1 0.6 0.6];
end

%% load embedding

%load('imagenet_val_embed.mat'); % load x (the embedding 2d locations from tsne)
points = bsxfun(@minus, points, min(points));
divFactor = max(points);
divFactor(divFactor==0) = 1; %%if only one point
points = bsxfun(@rdivide, points, divFactor);

%% load validation image filenames

N = length(imgNames);

%% create an embedding image

Sx = 5000; % size of full embedding image
Sy = 2600; % size of full embedding image

G = 255*ones(Sy, Sx, 3, 'uint8');
s = 200; % size of every single image
imFrac = 1;
%Ntake = 50000; ???
Ntake = size(points,1);

%% Embed
% 
% for i=1:Ntake
%     
%     if mod(i, 100)==0
%         fprintf('%d/%d...\n', i, Ntake);
%     end
%     % location
%     a = ceil(points(i, 1) * (S-s)+1);
%     b = ceil(points(i, 2) * (S-s)+1);
%     a = a-mod(a-1,s)+1;
%     b = b-mod(b-1,s)+1;
%     if G(a,b,1) ~= 0
%         continue % spot already filled
%     end
% 
% 	% load the image
%     
%     [imPath,imExt] = getDatasetImgDir(imgDatasets{i});
%     I = imread(fullfile(imPath,[imgNames{i} imExt]));
% 
%     bbox = boxes(i,:);
%     bbox(1:2) = max(bbox(1:2),1);
%     bbox(3) = min(bbox(3),size(I,2));bbox(4) = min(bbox(4),size(I,1));
%     I = I(bbox(2):bbox(4),bbox(1):bbox(3),:);
%         
% 	if size(I,3)==1, I = cat(3,I,I,I); end
%     I = imresize(I, [s, s]);
%     
%     G(a:a+s-1, b:b+s-1, :) = I;
%     
% end
% 
% imagesc(G);
% 
% %imwrite(G, [results_path result_file], 'jpg');

%% do a guaranteed quade grid layout by taking nearest neighbor

used = false(N, 1);

qq=length(1:s:Sx)*length(1:s:Sy);
abes = zeros(qq,2);
i=1;
for a=1:s:Sy
   for b=1:s:Sx
       abes(i,:) = [a,b];
       i=i+1;
   end
end
%abes = abes(randperm(size(abes,1)),:); % randperm

for i=1:size(abes,1)
   a = abes(i,1);
   b = abes(i,2);
   yf = (a-1)/Sy;
   xf = (b-1)/Sx;
   dd = sum(bsxfun(@minus, points, [yf, xf]).^2,2);
   dd(used) = inf; % dont pick these
   [dv,di] = min(dd); % find nearest image
    
   if(dv < 0.004)
       used(di) = true; % mark as done

       [imPath,imExt] = getDatasetImgDir(imgDatasets{di});
       I = imread(fullfile(imPath,[imgNames{di} imExt]));

        bbox = boxes(di,:);
        bbox(1:2) = max(bbox(1:2),1);
        bbox(3) = min(bbox(3),size(I,2));bbox(4) = min(bbox(4),size(I,1));
        I = I(bbox(2):bbox(4),bbox(1):bbox(3),:);

       if size(I,3)==1, I = cat(3,I,I,I); end
       
       if(correctInds(di))
           colorNum = 1;
       else
           colorNum = 2;
       end
       curr_color = visColormap(colorNum,:);
       I(:,:,1) = I(:,:,1)*curr_color(1);
       I(:,:,2) = I(:,:,2)*curr_color(2);
       I(:,:,3) = I(:,:,3)*curr_color(3);

       I = imresize(I, [s*imFrac, s*imFrac]);

       G(a:a+s*imFrac-1, b:b+s*imFrac-1, :) = I;
   end
   if mod(i,100)==0
       fprintf('%d/%d\n', i, size(abes,1));
   end
end

imagesc(G);

end
