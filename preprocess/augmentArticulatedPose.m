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
rightCoordSys = metadata.right_coordinate_sys{cInd};
rightCoordSys(1:6) = I(rightCoordSys(:));
rightHandNames = partNames(rightCoordSys);
[nKpSrt,kpSrtIds] = sort(cellfun(@(a) size(a,2), metadata.rigid_parts{cInd}), 'descend');
rigidKpInds = [];
counter = 1;
while (numel(rigidKpInds) < 7 && counter <= numel(nKpSrt))
    rigidKpInds = [rigidKpInds metadata.rigid_parts{cInd}{kpSrtIds(counter)}];
    counter = counter + 1;
end
rigidKpInds = unique(rigidKpInds);
rigidKpInds = sort(I(rigidKpInds));
rigidPartNames =  partNames(rigidKpInds);

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

%% sfm model computation and extracting rotations
disp('Learning SfM model')
[model, selectedInds] = learnSfmModel(dataStruct);
selectedInds = sort(find(selectedInds));


%% visualize sfm model

%visualizeWireframe(model.S,rigidPartNames,wireframeCar());
%keyboard;

% N = length(selectedInds)/2;
% quats = zeros(N,4);
% quatsMirror = zeros(N,4);
% 
% for n=1:N
%     quats(n,:) = dcm2quat(model.M{selectedInds(n)});
%     if(quats(n,1)<0)
%         quats(n,:) = -quats(n,:);
%     end
%     quatsMirror(n,:) = dcm2quat(model.M{selectedInds(n+N)});
%     if(quatsMirror(n,1)<0)
%         quatsMirror(n,:) = -quatsMirror(n,:);
%     end
% end

%keyboard;
%% right hand coordinate ambiguity

%% get canonical frame via quaternion optimization
% q = (v1,v2,v3,v4), q_mirror = (v1,v2,-v3,-v4)*(+/-1)
% z is vertical axis is the shape
% y axis along length of the car
% YZ plane is the symmetry plane with X increasing from object's anatomical
% right to left

leftInds = cellfun(@(x) ~isempty(x),strfind(rigidPartNames,'Left'));
leftInds = leftInds | cellfun(@(x) ~isempty(x),strfind(rigidPartNames,'L_'));
leftInds = leftInds | cellfun(@(x) ~isempty(x),strfind(rigidPartNames,'left'));
leftInds = find(leftInds);

kpsPerm = findKpsPerm(rigidPartNames);
rightInds = kpsPerm(leftInds);
Rx = alignSfmModel(model.S,[leftInds rightInds],I(metadata.horzEdges{cInd}),[]);

keyboard;


end