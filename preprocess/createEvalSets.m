function [goodInds] = createEvalSets(class,split)
%CREATEEVALSETS Summary of this function goes here
%   Detailed explanation goes here

%disp('Creating Eval Sets');

if(nargin<2)
    split = 1;
end

globals;

if(split)
    saveFile = fullfile(cachedir,'splitSets',class);
else
    saveFile = fullfile(cachedir,'evalSets',class);
end

%% Data for train & val sets

var = load(fullfile(cachedir,['rotationData' params.vpsDataset],class));
rotData = var.rotationData;
varFile = (fullfile(cachedir,['rcnnPredsVps' params.vpsDataset],params.features,[class '.mat']));
if(exist(varFile,'file'))
    var = load(varFile);
    featInd = find(ismember(var.outNames,'poseClassify'));
    feat = var.featStruct{featInd};
    feat = cell2mat(feat);
else
    disp('Warning : no features')
    feat = zeros(length(rotData),1);
end
rec_ids = vertcat(rotData.voc_rec_id);
bboxes = vertcat(rotData.bbox);
IoUs = vertcat(rotData.IoU);
%voc_ids = vertcat(rotData.voc_image_id);
if(isfield(rotData(1),'objectInd'))
    objectInds = vertcat(rotData.objectInd);
else
    objectInds = zeros(size(IoUs));
end
if(isfield(rotData(1),'dataset'))
    datasetNames = {rotData(:).dataset};
else
    datasetNames = repmat({'pascal'},length(rotData),1);
end
%eulers = horzcat(rotData.euler);eulers = eulers';
eulers = [];rots = [];goodInds = [];voc_ids = {};masks = {};views = {};
for i=1:length(rotData)
    if(~isempty(rotData(i).euler) && sum(rotData(i).euler == 0)~=3)
        goodInds(end+1)=i;
        rot = rotData(i).rot(:);
        rots(end+1,:)=rot';
        eulers(end+1,:) = rotData(i).euler';
        voc_ids{end+1} = rotData(i).voc_image_id;
        masks{end+1} = rotData(i).mask;
        views{end+1} = rotData(i).view;
    end
end
feat = feat(goodInds,:);
rec_ids = rec_ids(goodInds);
%voc_ids = voc_ids(goodInds);
bboxes = bboxes(goodInds,:);
IoUs = IoUs(goodInds);
objectInds = objectInds(goodInds);
datasetNames = datasetNames(goodInds);

%% Creating train, val, test partitions


%% Train
if(split)
    inds = ismember(voc_ids,sets.fnamesTrain);
    sets = load(fullfile(cachedir,['trainValTestSets' params.vpsDataset]));
else
    inds = true(length(voc_ids),1);
end
data.feat = feat(inds,:);
data.eulers = eulers(inds,:);
data.rots = rots(inds,:);
data.voc_ids = voc_ids(inds);
data.rec_ids = rec_ids(inds);
data.bboxes = bboxes(inds,:);
data.IoUs = IoUs(inds,:);
data.masks = masks(inds);
data.views = views(inds);
data.objectInds = objectInds(inds);
data.dataset = datasetNames(inds);
train = data;

if(~split)
    test = data;
    save(saveFile,'test','-v7.3');
    return;
end

%% Val
inds = ismember(voc_ids,sets.fnamesVal);
data.feat = feat(inds,:);
data.eulers = eulers(inds,:);
data.rots = rots(inds,:);
data.voc_ids = voc_ids(inds);
data.rec_ids = rec_ids(inds);
data.bboxes = bboxes(inds,:);
data.IoUs = IoUs(inds,:);
data.masks = masks(inds);
data.views = views(inds);
data.objectInds = objectInds(inds);
data.dataset = datasetNames(inds);
val = data;

%% Test
%inds = perm((Ntrain+Nval+1):end);
%inds = ismember(voc_ids,voc_ids_unique(inds));
inds = ismember(voc_ids,sets.fnamesTest);
data.feat = feat(inds,:);
data.eulers = eulers(inds,:);
data.rots = rots(inds,:);
data.voc_ids = voc_ids(inds);
data.rec_ids = rec_ids(inds);
data.bboxes = bboxes(inds,:);
data.IoUs = IoUs(inds,:);
data.masks = masks(inds);
data.views = views(inds);
data.objectInds = objectInds(inds);
test = data;

%% Save
save(saveFile,'train','val','test','-v7.3');

end

