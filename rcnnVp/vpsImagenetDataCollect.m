function [] = vpsImagenetDataCollect()

%RCNNIMAGENETDATACOLLECT Summary of this function goes here
%   Detailed explanation goes here


%% Initialize
globals;
mkdir(rcnnVpsImagenetDataDir);
delete([rcnnVpsImagenetDataDir '/*.mat']);

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
classes = {'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','plant','sheep','sofa','train','tvmonitor'};
classInds = [1 2 4 5 6 7 9 11 14 18 19 20];
classInds = [1 2 4 5 7 9 11 18 19 20]; %no motorbike and bus

for classInd = classInds
    class = classes{classInd};
    disp(class);
    rotationData = load(fullfile(cachedir,'rotationDataImagenet',class));
    rotationData = rotationData.rotationData;
    rotationData = rotationData(ismember({rotationData(:).dataset},'imagenet'));
    
    for n=1:length(rotationData)
        %disp([int2str(n) '/' int2str(length(rotationData))]);
        rcnnDataFile = fullfile(rcnnVpsImagenetDataDir,[rotationData(n).voc_image_id '.mat']);
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

end