function [] = augmentArticulatedPose(class)
%AUGMENTARTICULATEDPOSE Summary of this function goes here
%   Detailed explanation goes here

globals;
%% init
load(fullfile(cachedir,'vocKpMetadata'))
load(fullfile(rotationPascalDataDir,class));
if(~exist('pascalData','var'))
    pascalData = rotationData;
end
cInd = pascalClassIndex(class);
partNames = pascalData(1).part_names;
assert(length(metadata.kp_names{cInd})==length(pascalData(1).part_names));

%% Finding rigid parts and right hand coords sys
[~,I] = sort(partNames);

% I takes metadata partInds to current inds

[nKpSrt,kpSrtIds] = sort(cellfun(@(a) size(a,2), metadata.rigid_parts{cInd}), 'descend');
rigidKpInds = [];
counter = 1;
while (numel(rigidKpInds) < 7 && counter <= numel(nKpSrt))
    rigidKpInds = [rigidKpInds metadata.rigid_parts{cInd}{kpSrtIds(counter)}];
    counter = counter + 1;
end
rigidKpInds = unique(rigidKpInds);
rigidKpInds = sort(I(rigidKpInds));% is we have 5 parts, rigidKpInds is, say, [2 3 5]
rigidPartNames =  partNames(rigidKpInds);
rigidPerm = zeros(size(partNames));
rigidPerm(rigidKpInds) = 1:numel(rigidKpInds); % indexes normal part to its rigidInd
%keyboard;

%% add mirrored data
flipData = createFlippedDataPascal(pascalData,findKpsPerm(pascalData(1).part_names));

%% making dataStruct (voc_image_id,kps,bbox)
dataStruct.bbox = [];
dataStruct.kps = {};
dataStruct.voc_image_id = {};
for i=1:length(flipData)
    dataStruct.bbox(i,:) = flipData(i).bbox;
    dataStruct.kps{i} = flipData(i).kps(rigidKpInds,:);
    dataStruct.voc_image_id{i} = flipData(i).voc_image_id;
end

%% canonical directions vertices
leftInds = cellfun(@(x) ~isempty(x),strfind(rigidPartNames,'Left'));
leftInds = leftInds | cellfun(@(x) ~isempty(x),strfind(rigidPartNames,'L_'));
leftInds = leftInds | cellfun(@(x) ~isempty(x),strfind(rigidPartNames,'left'));
leftInds = find(leftInds);

kpsPerm = findKpsPerm(rigidPartNames);
rightInds = kpsPerm(leftInds);
lrEdges = [leftInds rightInds];
horzEdges = rigidPerm(I(metadata.horzEdges{cInd}));
vertEdges = rigidPerm(I(metadata.vertEdges{cInd}));

rightCoordSys = metadata.right_coordinate_sys{cInd};
rightCoordSys(1:6) = rigidPerm(I(rightCoordSys(:)));
rightHandNames = rigidPartNames(rightCoordSys);

if(strcmp(class,'bird')) %the mirror edges are screwed up !
    lrEdges = horzEdges;
    horzEdges = vertEdges;
    vertEdges = [];
end
%keyboard;

%% sfm model computation and extracting rotations
disp('Learning SfM model')
[model, selectedInds] = learnSfmModel(dataStruct);
selectedInds = sort(find(selectedInds));

%% get canonical frame via quaternion optimization
% q = (v1,v2,v3,v4), q_mirror = (v1,v2,-v3,-v4)*(+/-1)
% z is vertical axis is the shape
% y axis along length of the car
% YZ plane is the symmetry plane with X increasing from object's anatomical
% right to left

goodModel = 0;
flip = 0;
while(~goodModel)
    R = alignSfmModel(model.S,lrEdges,horzEdges,vertEdges);
    if(strcmp(class,'bird')) %the mirror edges are screwed up !
        R = [0 0 1;1 0 0;0 1 0]*R;
    end
    Srot = R*model.S;
    show3dModel(Srot,rigidPartNames,'convex_hull');
    userIn = input('Is this model aligned ? "y" will save and "n" will realign after flipping \n','s');
    if(strcmp(userIn,'y'))
        goodModel = 1;
        disp('Ok, saved !')
    else
        flip = mod(flip+1,2);
        model.S = diag([1 1 -1])*model.S;
        close all;
    end
end



end