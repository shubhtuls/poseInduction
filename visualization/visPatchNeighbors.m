function [] = visPatchNeighbors(rotationData,featIm,imBox,feat)
%VISPATCHNEIGHBORS Summary of this function goes here
%   Detailed explanation goes here

globals;

%% selecting patch
fSize = size(feat{1});
imagesc(imBox);
[x,y] = ginput(1);
close all;
xInd = round(fSize(1)*x/size(imBox,2)+0.5);
yInd = round(fSize(2)*y/size(imBox,1)+0.5);
featPatch = featIm(xInd,yInd,:);

%% visualization
numNeighbors = 59;
plotH = 6;
plotW = 10;

dists = cellfun(@(x) {sum(abs(bsxfun(@minus,x,featPatch)),3)},feat);
dist = cellfun(@(x) min(min(x)),dists);
[~,p] = sort(dist,'ascend');

for i = 1:(numNeighbors+1)
    ind = p(i);
    [~,minInd] = min(dists{ind}(:));[xInd,yInd] = ind2sub(fSize(1:2),minInd);
    subplot(plotH,plotW,i);
    
    [imgDir,imgExt] = getDatasetImgDir(rotationData(ind).dataset);
    im = imread(fullfile(imgDir,[rotationData(ind).voc_image_id imgExt]));
    
    bbox = rotationData(ind).bbox;
    xDiff = (bbox(3)-bbox(1))/fSize(1);
    yDiff = (bbox(4)-bbox(2))/fSize(2);         
    bbox = [bbox(1) + (xInd-2.0)*xDiff,bbox(2) + (yInd-2.0)*yDiff,bbox(1) + (xInd+1)*xDiff,bbox(2) + (yInd+1)*yDiff];
    bbox = round(bbox);
    
    bbox(1:2) = max(bbox(1:2),1);
    bbox(3) = min(size(im,2),bbox(3));
    bbox(4) = min(size(im,1),bbox(4));
    imagesc(im(bbox(2):bbox(4),bbox(1):bbox(3),:));
    axis image;axis off;
    %showboxes(im,dataStruct.bbox(ind,:));
end

%keyboard;
%[~,p] = sort(dist,'ascend');


end

function f = sigmoid(f)
    f = 1./(1+exp(-f));
end