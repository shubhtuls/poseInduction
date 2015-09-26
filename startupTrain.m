globals;
if exist(fullfile(cachedir,'pascalTrainValIds.mat'))
    load(fullfile(cachedir,'pascalTrainValIds'))
else
    fIdTrain = fopen(fullfile(pascalDir,'VOC2012','ImageSets','Main','train.txt'));
    trainIds = textscan(fIdTrain,'%s');
    trainIds = trainIds{1};
    fIdVal = fopen(fullfile(pascalDir,'VOC2012','ImageSets','Main','val.txt'));
    valIds = textscan(fIdVal,'%s');
    valIds = valIds{1};
    save(fullfile(cachedir,'pascalTrainValIds.mat'),'trainIds','valIds');
end

if ~exist(fullfile(cachedir,'imagenetTrainIds.mat'))
    fnamesTrain = generateImagenetTrainNames();
    save(fullfile(cachedir,'imagenetTrainIds.mat'),'fnamesTrain');
end

if(~exist(fullfile(caffeModelDir,'vgg','netFinal.caffemodel'))
    mkdirOptional(fullfile(caffeModelDir,'vgg'))
    websave(fullfile(caffeModelDir,'vgg','netFinal.caffemodel'),'http://www.cs.berkeley.edu/~shubhtuls/cachedir/poseInduction/snapshots/vgg/netFinal.caffemodel')
end