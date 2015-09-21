function [] = vpsPascalDataCollect(suffix)
%RCNNDATA Summary of this function goes here
%   Detailed explanation goes here

%% Initialize
globals;
if(nargin<1)
    suffix = '';
end
globals;
mkdirOptional(rcnnVpsPascalDataDir);
delete([rcnnVpsPascalDataDir '/*.mat']);
%candidateThresh = params.candidateThresh;
candidateThresh = 0.5; %more filtering done later in rcnnTrainValTestCreate

%% WINDOW FILE FORMAT
% repeated :
%   img_path(abs)
%   reg2sp file path
%   num_pose_param
%   channels
%   height
%   width
%   num_windows
%   classIndex overlap x1 y1 x2 y2 regionIndex poseParam0 .. poseParam(numPoseParam)

%% Format of file saved
% for each image, same file containing -
% classIndex overlap bbox regionIndex eulers


%% Iterate over classes
classes = {'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'};
%classInds = [1 2 4 5 6 7 9 11 14 18 19 20];
classInds = [1 2 4 5 7 9 11 18 19 20]; %no motorbike and bus
articulatedInds = [3 8 10 13 15 16]; %no sheep and dog (12,17)

% no bottles
%classInds = [1 2 4 6 7 9 11 14 18 19 20];
rotationDatas = {};
for c = [articulatedInds classInds]
    class = classes{c};
    disp(class);
    rotationData = load(fullfile(cachedir,['rotationDataJoint' suffix],class));
    rotationDatas{c} = rotationData.rotationData;
end

load(fullfile(cachedir,'pascalTrainValIds'));
fnames = [trainIds;valIds];

for classInd = [articulatedInds classInds]
    class = classes{classInd};
    rotationData = rotationDatas{classInd};
    objInds = find(ismember({rotationData(:).dataset},'pascal'));
    
    for n=objInds
        %disp([int2str(n) '/' int2str(length(rotationData))]);
        rcnnDataFile = fullfile(rcnnVpsPascalDataDir,[rotationData(n).voc_image_id '.mat']);
        bbox = overlappingBoxes(rotationData(n).bbox,rotationData(n).imsize);        
        nCands = size(bbox,1);
        
        classIndex = classInd*ones(nCands,1);
        overlap = ones(nCands,1);
        regionIndex = zeros(nCands,1);
        euler = repmat(rotationData(n).euler',nCands,1);
        imSize = rotationData(n).imsize;
        
        %% Saving
        if(~isempty(classIndex))
            if(exist(rcnnDataFile,'file'))
                rcnnData = load(rcnnDataFile);
                overlap = vertcat(rcnnData.overlap,overlap);
                euler = vertcat(rcnnData.euler,euler);
                bbox = vertcat(rcnnData.bbox,bbox);
                classIndex = vertcat(rcnnData.classIndex,classIndex);
                regionIndex = vertcat(rcnnData.regionIndex,regionIndex);
            end
            save(rcnnDataFile,'overlap','euler','bbox','classIndex','regionIndex','imSize');
        end
    end
end

% for n=1:length(fnames)
%     disp([int2str(n) '/' int2str(length(fnames))]);
%     candFile = fullfile(pascalCandsDir,[fnames{n} '.mat']);
%     rcnnDataFile = fullfile(rcnnVpsPascalDataDir,[fnames{n} '.mat']);
%     
%     maskIndices = [];
%     if(~exist(candFile,'file'))
%         continue;
%     end
%     cands = load(candFile);
%     
%     overlap = [];
%     euler = [];
%     bbox = [];
%     classIndex = [];
%     regionIndex = [];
%     
%     for c = [articulatedInds classInds]
%         rotationData = rotationDatas{c};
%         objInds = find(ismember({rotationData(:).voc_image_id},fnames(n)));
%         for o = objInds
%             oBbox = rotationData(o).bbox;
%             
%             candsOverlap = IoUs(oBbox,cands.bbox);
%             goodCands = candsOverlap >= candidateThresh & (cands.class == 0 | cands.class == c) & ismember(cands.method',{'mcg','gt'});
%             maskIndices = find(goodCands);
%             if(~isempty(maskIndices))
%                 nCands = numel(maskIndices);
%                 classIndex = vertcat(classIndex,c*ones(length(maskIndices),1));
%                 overlap = vertcat(overlap,candsOverlap(maskIndices));
%                 bbox = vertcat(bbox,cands.bbox(maskIndices,:));
%                 regionIndex = vertcat(regionIndex,maskIndices);
%                 euler = vertcat(euler,repmat(rotationData(o).euler',nCands,1));
%             end
%         end
%     end
% 
%     %% Saving
%     if(~isempty(classIndex))
%         save(rcnnDataFile,'overlap','euler','bbox','classIndex','regionIndex');
%     end
% 
% end

end


function o = IoUs(b,a)

x1 = max(a(:,1), b(1));
y1 = max(a(:,2), b(2));
x2 = min(a(:,3), b(3));
y2 = min(a(:,4), b(4));

w = x2-x1+1;
h = y2-y1+1;
inter = w.*h;
aarea = (a(:,3)-a(:,1)+1) .* (a(:,4)-a(:,2)+1);
barea = (b(3)-b(1)+1) * (b(4)-b(2)+1);
% intersection over union overlap
o = inter ./ (aarea+barea-inter);
% set invalid entries to 0 overlap
o(w <= 0) = 0;
o(h <= 0) = 0;

end
