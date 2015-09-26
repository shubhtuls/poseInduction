global pascalDir
global cachedir
global PASCAL3Ddir
global rcnnVpsPascalDataDir
global rcnnVpsImagenetDataDir
global rcnnVpsJointDataDir
global rotationPascalDataDir
global rotationImagenetDataDir
global rotationIlsvrcDataDir
global rotationJointDataDir
global params
global pascalImagesDir
global imagenetImagesDir
global segkpAnnotationDir
global caffeModelDir
global caffeProtoDir
global basedir

folders = {'encoding','evaluate','utils','visualization','evaluation','learning','preprocess','rcnnKp','rcnnVp','cnnFeatures','sfm','optimization'};

for i=1:length(folders)
    addpath(genpath(folders{i}));
end

basedir = pwd();
cachedir  = fullfile(basedir,'cachedir'); % directory where all the intermediate computations and data will be saved

%% These paths might need to be edited

PASCAL3Ddir = fullfile(basedir,'data','PASCAL3D');
pascalDir = fullfile(basedir,'data','VOCdevkit');
pascalImagesDir = fullfile(basedir,'data','VOCdevkit','VOC2012','JPEGImages');
imagenetImagesDir = fullfile(basedir,'data','imagenet','images');
rcnnDetectionsFile = fullfile(basedir,'data','VOC2012_val_det.mat');
segkpAnnotationDir = fullfile(basedir,'data','segkps'); %required for keypoint prediction
caffeModelDir = fullfile(cachedir,'snapshots'); %directory where caffemodels are saved - you'll have to set this up

%cachedir  = '/work5/shubhtuls/cachedir/poseInduction/';
%pascalDir = '/work5/shubhtuls/cachedir/Datasets/VOCdevkit/';
%PASCAL3Ddir = '/work5/shubhtuls/cachedir/Datasets/PASCAL3D';
%pascalCandsDir = '/work5/shubhtuls/cachedir/Datasets/pascalCandidates/candidateAnnotations/';
%pascalImagesDir = '/work5/shubhtuls/cachedir/Datasets/VOCdevkit/VOC2012/JPEGImages/';

%imagenetImagesDir = '/work5/shubhtuls/cachedir/Datasets/imagenet/images/';
%imagenetDir = '/work5/shubhtuls/cachedir/Datasets/imagenet/';

%ilsvrcImagesDir = '/work5/shubhtuls/cachedir/Datasets/ilsvrc13/ILSVRC2013_DET_train/';
%ilsvrcDir = '/work5/shubhtuls/cachedir/Datasets/ilsvrc13/';

%annotationDir = '/work5/shubhtuls/cachedir/Datasets/pascalAnnotations';
%candsDir = '/work5/shubhtuls/cachedir/Datasets/pascalCandidates/candidateAnnotations/';

params = getParams;

rcnnVpsPascalDataDir = fullfile(cachedir,'rcnnVpsPascalData');mkdirOptional(rcnnVpsPascalDataDir);
rcnnVpsImagenetDataDir = fullfile(cachedir,'rcnnVpsImagenetData');mkdirOptional(rcnnVpsImagenetDataDir);
rcnnVpsJointDataDir = fullfile(cachedir,'rcnnVpsJointData');mkdirOptional(rcnnVpsJointDataDir);

rotationPascalDataDir = fullfile(cachedir,'rotationDataPascal');mkdirOptional(rotationPascalDataDir);
rotationImagenetDataDir = fullfile(cachedir,'rotationDataImagenet');mkdirOptional(rotationImagenetDataDir);
rotationIlsvrcDataDir = fullfile(cachedir,'rotationDataIlsvrc');mkdirOptional(rotationIlsvrcDataDir);
rotationJointDataDir = fullfile(cachedir,'rotationDataJoint');mkdirOptional(rotationJointDataDir);

%websiteDir = '/work5/shubhtuls/website/public_html/visualization/poseInduction';
%caffeModelDir = '/work5/shubhtuls/snapshots/poseInduction/finalSnapshots/';
%caffeProtoDir = '/work5/shubhtuls/prototxts/poseInduction/';
caffeProtoDir = fullfile(basedir,'prototxts');

%imdbIlsvrc = load(fullfile(ilsvrcDir,'imdbs','imdb_ilsvrc13_val'));imdbIlsvrc = imdbIlsvrc.imdb;
%roidbIlsvrc = load(fullfile(ilsvrcDir,'roidbs','roidb_ilsvrc13_val'));roidbIlsvrc = roidbIlsvrc.roidb;

clear i;
clear folders;

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

if exist('external/caffe/matlab/+caffe')
  addpath('external/caffe/matlab/');
else
  warning('Please install Caffe in ./external/caffe');
end
