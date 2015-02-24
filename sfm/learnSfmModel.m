function [model,goodInds] = learnSfmModel(dataStruct)
%LEARNSFMMODEL Summary of this function goes here
%   Detailed explanation goes here

goodInds = cellfun(@(x) ~isempty(x) && (sum(~isnan(x(:)))>=6),dataStruct.kps); %atleast 3 visible keypoints


D = size(dataStruct.kps{1},1);

for i=1:length(dataStruct)
    bbox = dataStruct.bbox(i,:);
    maxDim = max(bbox([4 3]) - bbox([2 1]));
    dataStruct.kps{i} = (dataStruct.kps{i} - repmat(bbox([3 4]) + bbox([1 2]),D,1)/2)*10/maxDim;
end

P = (horzcat(dataStruct.kps{goodInds}));
P = P';

[M , S, ~] = sfmFactorization(P, 50, 10);

%keyboard;
rots = {};
ctr = 1;
for i=1:numel(goodInds)
    if(goodInds(i))
        rot = M(2*ctr+[-1 0],:);
        rot = rot/norm(rot(1,:));
        rot = [rot;cross(rot(1,:),rot(2,:))];
        if(det(rot)<0)
            rot(3,:) = -rot(3,:);
        end
        rots{i} = rot;
        ctr = ctr+1;
    end
end

model.S = S;
model.M = rots;

end

