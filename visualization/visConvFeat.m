function [] = visConvFeat(data,feat,inds)
%VISCONVFEAT Summary of this function goes here
%   Detailed explanation goes here

C = size(feat{inds(1)},3);
numChannels = min(C,50);

figure(1);

for j=1:length(inds)
    subplot(2,1,j);
    i = inds(j);
    bbox = data.bboxes(i,:);
    [imgDir,imgExt] = getDatasetImgDir(data.dataset{i});
    im = imread(fullfile(imgDir,[data.voc_ids{i} imgExt]));
    bbox(1:2) = max(bbox(1:2),1);
    bbox(3) = min(bbox(3),size(im,2));bbox(4) = min(bbox(4),size(im,1));
    imBox = im(bbox(2):bbox(4),bbox(1):bbox(3),:);
    imagesc(imBox);hold on;
end

figure(2);
plotH = 10;
plotW = 10;
cInds = randperm(C,numChannels);
for c = 1:numChannels
    f1 = squeeze(feat{inds(1)}(:,:,cInds(c)));
    f2 = squeeze(feat{inds(2)}(:,:,cInds(c)));
    subplot(plotH,plotW,2*c-1)
    imagesc(f1);
    
    subplot(plotH,plotW,2*c)
    imagesc(f2);
end

end