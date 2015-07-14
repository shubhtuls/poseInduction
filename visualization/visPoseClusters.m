function [] = visPoseClusters(azimuths, imgNames, imgDatasets, boxes, topInds, saveDir)
%VISPOSECLUSTERS Summary of this function goes here
%   Detailed explanation goes here

if(nargin < 6)
    saveDir = [];
end
N = numel(azimuths);
topFractions = [1/3 1/2 2/3 1];

coverage = pi*[-1/2,1/2];
binSize = 1/4;

binCentres = pi*[(-1+binSize):binSize:(1-binSize)];
nBins = numel(binCentres);

coveredBins = binCentres((binCentres >= coverage(1)) & (binCentres <= coverage(2)));
nCoveredBins = sum((binCentres >= coverage(1)) & (binCentres <= coverage(2)));

binH = 2;
binW = 3;
numPerBin = binH*binW;
instanceInds = {};
for b = 1:nBins
    for t = topFractions
        goodInds = topInds(1:round(N*t));
        binInds = find(abs(azimuths(goodInds) - binCentres(b))<binSize);
        if(numel(binInds) >= numPerBin)
            break;
        end
    end
    instanceInds{b} = topInds(binInds);
end

plotH = 1;
plotW = ceil(nCoveredBins/plotH);

imSize = 100;

bCount = 0;
for b = find((binCentres >= coverage(1)) & (binCentres <= coverage(2)))
    bCount = bCount+1;
    binIm = uint8(255*ones(imSize*binH,imSize*binW,3));
    numImgs = min(length(instanceInds{b}),numPerBin);
    count = 1;
    inds = randperm(length(instanceInds{b}),numImgs);
    for j = inds
        W = mod(count,binW)+1;
        H = ceil(count/binW);
        count = count+1;
        
        di = instanceInds{b}(j);
        [imPath,imExt] = getDatasetImgDir(imgDatasets{di});
        I = imread(fullfile(imPath,[imgNames{di} imExt]));
        
        bbox = boxes(di,:);
        bbox(1:2) = max(bbox(1:2),1);
        bbox(3) = min(bbox(3),size(I,2));bbox(4) = min(bbox(4),size(I,1));
        I = I(bbox(2):bbox(4),bbox(1):bbox(3),:);
        if size(I,3)==1, I = cat(3,I,I,I); end
        
        binIm(imSize*(H-1) + [1:imSize],imSize*(W-1) + [1:imSize],:) = imresize(I,[imSize imSize]);
        if(saveDir)
            imwrite(binIm,fullfile(saveDir,[num2str(bCount) '.png']));
        end
        subplot(plotH,plotW,bCount);
        imagesc(binIm);axis image;
    end
    
end


end