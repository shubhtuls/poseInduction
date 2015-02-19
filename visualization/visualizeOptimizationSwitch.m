function [] = visualizeOptimizationSwitch(prediction,encoding,cls,voc_id,dataset,bbox,objectInd,scores)
%VISUALIZEOPTIMIZATIONSWITCH Summary of this function goes here
%   Detailed explanation goes here

globals;
CADPath = fullfile(PASCAL3Ddir,'CAD',cls);
cad = load(CADPath);
cad = cad.(cls);
eulersPred = decodePose(prediction,encoding);

%% pascal 3d
pascal3Dfile = fullfile(PASCAL3Ddir,'Annotations',[cls '_' dataset],[voc_id '.mat']);
record = load(pascal3Dfile);record = record.record;
viewpoint = record.objects(objectInd).viewpoint;
vertex = cad(record.objects(objectInd).cad_index).vertices;
face = cad(record.objects(objectInd).cad_index).faces;

%% image
subplot(3,1,1);
[imgDir,imgExt] = getDatasetImgDir(dataset);
im = imread(fullfile(imgDir,[voc_id imgExt]));
bbox(1:2) = max(bbox(1:2),1);
bbox(3) = min(bbox(3),size(im,2));bbox(4) = min(bbox(4),size(im,1));
imBox = im(bbox(2):bbox(4),bbox(1):bbox(3),:);
imagesc(imBox);hold on;

%% visualize new and old rot

for i=1:2
    viewpoint.azimuth = eulersPred(i,3);
    viewpoint.elevation = eulersPred(i,2);
    viewpoint.theta = eulersPred(i,1);
    record.objects(objectInd).viewpoint = viewpoint;

    [x2d,Z] = project3d(vertex, record.objects(objectInd),face);
    axis equal;

    subplot(3,1,i+1);
    trisurf(face,x2d(:,1),-x2d(:,2),-Z);axis equal; %empirically found this view and plot to work best
    view(0,90);
    title(scores(i));
end

% pause();close all;
% disp('blah')

end

