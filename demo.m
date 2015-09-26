function [] = demo()
startup;

%% Download caffe model trained on all pascal classes
if ~exist(fullfile(caffeModelDir,'vggCommon20','netFinal.caffemodel'))
    mkdirOptional(fullfile(caffeModelDir,'vggCommon20'))
    system(sprintf('wget -P %s %s',fullfile(caffeModelDir,'vggCommon20'),'http://www.cs.berkeley.edu/~shubhtuls/cachedir/poseInduction/snapshots/vggCommon20/netFinal.caffemodel'))
end

%% load model in caffe
proto = 'vggCommon20';suffix = 'vggCommon20';
protoFile = fullfile(caffeProtoDir,proto,'deploy.prototxt');
binFile = fullfile(caffeModelDir,suffix, 'netFinal.caffemodel');
cnn_model=cnn_create_model(protoFile,binFile);
cnn_model=cnn_load_model(cnn_model);
cnn_model.input_size = 224;

meanNums = [102.9801,115.9465,122.7717]; %imagenet mean
for i=1:3
    meanIm(:,:,i) = ones(224)*meanNums(i);
end

cnn_model.image_mean = single(meanIm);
cnn_model.batch_size=20;

%% compute pose
demoData = makeDemoData();
featStruct = cnnFeaturesSingleBox(demoData, cnn_model,0,1);
featPose = featStruct{1};
caffe.reset_all()
preds = poseHypotheses(vertcat(featPose{:}),1,0);
azimuths = preds{1}(:,3);
%keyboard;

%% Display
for i=1:length(demoData.voc_image_id)
    view = azToView(azimuths(i));
    im = imread(fullfile(demoData.imgDir{i},[demoData.voc_image_id{i} demoData.imgExt{i}]));
    imagesc(im);axis image;
    title(sprintf('Object is %s facing',view));
    pause();close all;
end

end

function demoData = makeDemoData()
    globals;
    demoDir = fullfile(basedir,'demoIms');
    imgNames = getFileNamesFromDirectory(demoDir,'type',{'.jpg'});
    
    demoData = struct;
    demoData.voc_image_id = {};
    demoData.bbox = [];
    demoData.labels= [];
    demoData.imgExt = {};
    demoData.imgDir={};
    
    for i = 1:length(imgNames)
        im = imread(fullfile(demoDir,imgNames{i}));
        demoData.voc_image_id{i} = imgNames{i}(1:end-4);
        demoData.imgExt{i} = imgNames{i}(end-3:end);
        demoData.bbox(i,:) = [1 1 size(im,2) size(im,1)];
        demoData.imgDir{i} = demoDir;
        demoData.labels(i)  = 0; %would have needed index of most similar class for SCT
    end
    
end