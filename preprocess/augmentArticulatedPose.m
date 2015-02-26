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
    end
    close all;
end

%% computing rotation matrices
%if(flip)
%    refRot = diag([-1 -1 1]);
%else
%    refRot = diag([1 1 1]);
%end
%refRot = eye(3);
%rots = model.M;
%for i=1:length(rots)
%    if(~isempty(rots{i}))
%        rots{i} = refRot*rots{i}*R';
%    end
%end
model.S = Srot;
rots = fitSfmModel(dataStruct,model);


%% compute euler angles
%euler = rotationData(i).euler;
%rots{i}'*angle2dcm(euler(3),-euler(2)-pi/2,euler(1),'ZXZ')
%%% compute angles for inverse matrix
% [e1,e2,e3] = dcm2angle(rots{i}','ZXZ');if(e3>0) e3 = e3-2*pi; end 
% e1 = -e1;e2 = e2-pi/2;e3 = -e3;
% [e1;e2;e3] =  euler

for i=1:length(rotationData)
    if(~isempty(rots{i}))
         %compute euler angles for inverse matrix gives negative of desired
         % angles in inverse order. Note : I hate euler angles !
        [e1,e2,e3] = dcm2angle(rots{i}','ZXZ');
        if(e3>0) 
            e3 = e3-2*pi;
        end
        e1 = -e1;e2 = e2-pi/2;e3 = -e3;
        pascalData(i).eulersSfm = [e1;e2;e3];
    else
        pascalData(i).eulersSfm = [];
    end
end

%% Looking at similarity with p3d eulers
keyboard;

end