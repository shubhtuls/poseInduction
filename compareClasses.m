cInd1 = 7;
cInd2 = 6;

%% load / preprocess
load(fullfile(rotationJointDataDir,pascalIndexClass(cInd1)));
rotationData1 = rotationData;
load(fullfile(rotationJointDataDir,pascalIndexClass(cInd2)));
rotationData2 = rotationData;

var1 = load(fullfile(cachedir,'rcnnPredsVpsJoint','vggPool5',pascalIndexClass(cInd1)));
var2 = load(fullfile(cachedir,'rcnnPredsVpsJoint','vggPool5',pascalIndexClass(cInd2)));

%% reshape
for i=1:length(var1.feat)
    tmp = reshape(sigmoid(var1.feat{i}),[7 7 512]);
    var1.feat{i} = tmp;
end

for i=1:length(var2.feat)
    tmp = reshape(sigmoid(var2.feat{i}),[7 7 512]);
    var2.feat{i} = tmp;
end

%% visualize

for ind = randperm(numel(rotationData1))
    [imgDir,imgExt] = getDatasetImgDir(rotationData1(ind).dataset);
    im = imread(fullfile(imgDir,[rotationData1(ind).voc_image_id imgExt]));
    bbox = rotationData1(ind).bbox;
    im = (im(bbox(2):bbox(4),bbox(1):bbox(3),:));
    visPatchNeighbors(rotationData2,var1.feat{ind},im,var2.feat);
    pause();close all;
end