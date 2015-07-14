function [] = rcnnSimilarityJointTrainValTestCreate(suffix)
%RCNNTRAINVALTESTCREATE Summary of this function goes here
%   Detailed explanation goes here

%% WINDOW FILE FORMAT
% repeated :
%   img_path(abs)
%   reg2sp file path
%   sp file path
%   num_pose_param
%   channels
%   height
%   width
%   num_windows
%   classIndex overlap x1 y1 x2 y2 regionIndex poseParam0 .. poseParam(numPoseParam)

%% Initialization
globals;
params = getParams();

%% Train/Val/Test filenames generate

load(fullfile(cachedir,'jointTrainValTestSets.mat'));
load(fullfile(cachedir,'pascalTrainValIds.mat'));
finetuneDir = fullfile(cachedir,'rcnnFinetuneVps','similarityJoint');
mkdirOptional(finetuneDir);
fnamesSets = {};
fnamesSets{1} = unique([trainIds' fnamesTrain]); %train on pascal+imagenet
fnamesSets{2} = valIds';

%% Generating test files
%sets = {'Train','Val','Test'};
sets = {'Train','Val'};

for s=1:length(sets)
    set = sets{s};
    disp(['Generating data for ' set]);
    txtFile = fullfile(finetuneDir,[set suffix '.txt']);
    fid = fopen(txtFile,'w+');
    fnames = fnamesSets{s};
    count = 0;
    for j=1:length(fnames)
    %for j=1:1
        id = fnames{j};
        if(exist(fullfile(rcnnVpsPascalDataDir,[id '.mat']),'file'))
            candFile = fullfile(rcnnVpsPascalDataDir,[id '.mat']);
            dataset = 'pascal';
        elseif (exist(fullfile(rcnnVpsImagenetDataDir,[id '.mat']),'file'))
            candFile = fullfile(rcnnVpsImagenetDataDir,[id '.mat']);
            dataset = 'imagenet';
        else
            continue;
        end
        cands = load(candFile);
        if(isempty(cands.overlap))
            continue;
        end
        numcands = round(sum(cands.overlap >= params.candidateThresh));
        if(numcands ==0)
            continue;
        end
        count=count+1;
        %%%%%%%%%%%% Insert anakin paths here
        if(strcmp(dataset,'imagenet'))
            imgFile = ['/data1/shubhtuls/cachedir/imagenet/images/' id '.jpg'];
            imsize = cands.imSize;
        else
            imgFile = ['/data1/shubhtuls/cachedir/VOCdevkit/VOC2012/JPEGImages/' id '.jpg'];
            tmp = load(fullfile(pascalCandsDir,id));imsize = size(tmp.sp);
        end

        %fprintf(fid,'# %d\n%s\n%s\n%s\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n%d\n',count-1,imgFile,reg2spFile,spFile,3,10,10,20,3,imsize(1),imsize(2),numcands);
        fprintf(fid,'# %d\n%s\n%d\n%d\n%d\n%d\n',count-1,imgFile,3,imsize(1),imsize(2),numcands);
        %if(max(cands.euler(:,1))>=pi/2 || max(cands.euler(:,2)>=pi/2 ))
        %    disp('Oops');
        %end
        cands.quat = encodePose(cands.euler,'quat');
        for n=1:size(cands.overlap,1)
            
            fprintf(fid,'%d %f %d %d %d %d %0.5f %0.5f %0.5f %0.5f\n',...
                cands.classIndex(n),cands.overlap(n),...
                cands.bbox(n,1),cands.bbox(n,2),cands.bbox(n,3),cands.bbox(n,4),...
                cands.quat(n,1),cands.quat(n,2),cands.quat(n,3),cands.quat(n,4));
        end
    end
end

end