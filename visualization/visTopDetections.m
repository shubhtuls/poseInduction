function [] = visTopDetections(class)
%VISTOPDETECTIONS Summary of this function goes here
%   Detailed explanation goes here

globals;
candsDir = '/work5/shubhtuls/cachedir/poseRegression/detectionPose/Nms/';
load([candsDir 'allDets.mat']);
cInd = pascalClassIndex(class);
cands = dataStructs(cInd);

fid = fopen('../Datasets/VOCdevkit/VOC2012/ImageSets/Main/val.txt');
fnames = textscan(fid, '%s');
fnames = fnames{1};
cands.voc_ids = cell(size(cands.boxes));

CADPath = fullfile(PASCAL3Ddir,'CAD',class);
cad = load(CADPath);
cad = cad.(class);
vertices = cad(1).vertices;
faces = cad(1).faces;

for i=1:length(fnames)
    cands.voc_ids{i} = fnames(i*ones(size(cands.boxes{i},1),1));
end

boxes = vertcat(cands.boxes{:});
scores = boxes(:,5);
[scores,perm] = sort(scores,'descend');

boxes = boxes(perm,1:4);
feat = vertcat(cands.feat{:});
feat = feat(perm,:);
ids = vertcat(cands.voc_ids{:});
ids = ids(perm);

preds = poseHypotheses(feat,1,0);
preds = preds{1};
eulersPred = decodePose(preds,params.angleEncoding);
rotX = diag([1 -1 -1]);

%for i=1:length(ids)
for i = randperm(100)
    im = imread([PASCAL_DIR '/' ids{i} '.jpg']);
    subplot(1,2,1);
    showboxes(im,boxes(i,:));
    axis image;
    subplot(1,2,2);
    
    euler = eulersPred(i,:);
    R = angle2dcm(euler(1), euler(2)-pi/2, -euler(3),'ZXZ');
    R = rotX*R';
    verticesP = R*vertices';
    verticesP = verticesP';
    trisurf(faces,verticesP(:,1),verticesP(:,2),verticesP(:,3));axis equal;view(0,-90);
    title(num2str(scores(i)));    
    pause(1);close all;
end

end

