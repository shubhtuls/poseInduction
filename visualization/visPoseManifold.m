function [] = visPoseManifold(preds,predScores,data,poseBins,binSize)
%VISPOSEMANIFOLD Summary of this function goes here
%   Detailed explanation goes here

plotH = 3;
plotW = numel(poseBins);
preds = mod(preds*10.5/pi+9.5,21)+1;

for b = 1:plotW
    binInds = find(abs(preds(:,3)-poseBins(b)) <= binSize);
    [~,binPerm] = sort(predScores(binInds),'descend');
    binInds = binInds(binPerm);
    for i = 1:min(plotH,numel(binInds))
        j = binInds(i);
        subplot(plotH,plotW,(i-1)*plotW+b);
        [imgDir,imgExt] = getDatasetImgDir(data.dataset{j});
        im = imread(fullfile(imgDir,[data.voc_ids{j} imgExt]));

        bbox = data.bboxes(j,:);
        bbox(1:2) = max(bbox(1:2),1);
        bbox(3) = min(bbox(3),size(im,2));bbox(4) = min(bbox(4),size(im,1));
        imBox = im(bbox(2):bbox(4),bbox(1):bbox(3),:);

        imagesc(imBox);axis image;
    end
end

end