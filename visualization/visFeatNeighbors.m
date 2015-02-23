function [] = visFeatNeighbors(data,featSimilarInds,variableIndex)
%VISFEATNE Summary of this function goes here
%   Detailed explanation goes here

globals;
plotH = 10;
plotW = 7;

featSimilarInds = [variableIndex featSimilarInds'];
for i=1:numel(featSimilarInds)
    j = featSimilarInds(i);
    subplot(plotH,plotW,i);
    [imgDir,imgExt] = getDatasetImgDir(data.dataset{j});
    im = imread(fullfile(imgDir,[data.voc_ids{j} imgExt]));

    bbox = data.bboxes(j,:);
    bbox(1:2) = max(bbox(1:2),1);
    bbox(3) = min(bbox(3),size(im,2));bbox(4) = min(bbox(4),size(im,1));
    imBox = im(bbox(2):bbox(4),bbox(1):bbox(3),:);
    
    imagesc(imBox);

end

end