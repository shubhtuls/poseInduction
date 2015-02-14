function [inlierIndices,projPoints] = sfmRansac(kpsPred,sfmModel,thresh)
%SFMRANSAC Summary of this function goes here
%   Detailed explanation goes here


sfmIter = 10;
numSfmKps = 5;
if(size(kpsPred,2)==2)
    kpsPred = kpsPred'; %becomes 2 X N
end

nKp = size(kpsPred,2);

ransacIter = 1000;
predInds = find(~isnan(kpsPred(1,:)));

if(numel(predInds)<=numSfmKps)
    inlierIndices = predInds;
    return;
end

P = nan(2*ransacIter,nKp);
for i = 1:ransacIter
    thisInds = randperm(numel(predInds),numSfmKps);
    P(2*i+[-1 0],predInds(thisInds)) = kpsPred(:,predInds(thisInds));
end

[~, ~, W] = sfmFactorizationKnownShape(P, sfmModel.S, sfmIter);

maxInliers = 0;
inlierIndices = [];
projPoints = [];

for i=1:ransacIter
    sfmPred = W(2*i+[-1 0],:);
    diffs = sfmPred(:,predInds) - kpsPred(:,predInds);
    inliers = predInds(sqrt(sum(diffs.^2,1))<=thresh);
    if(numel(inliers) > maxInliers)
        maxInliers = numel(inliers);
        inlierIndices = inliers;
        projPoints = sfmPred;
    end
    
end

end