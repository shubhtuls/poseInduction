function [perfAblations, perfConfAblations] = mainExperiment()
%MAINEXPERIMENT Summary of this function goes here
%   Detailed explanation goes here
globals;
params = getParams();
classInds = [6 14 12 17]; %bus, motorbike, dog, sheep
%classInds = [17]; %bus, motorbike, dog, sheep
nClasses = length(classInds);
%usePascalViews = [1]; %evaluation using pascal3d or pascal voc labels
usePascalViews = [0 0 1 1]; %evaluation using pascal3d or pascal voc labels

%% Run SCT vs GC Experiment
perfAblations = zeros(5,nClasses);
perfConfAblations = zeros(3,nClasses);

params.features = 'vggJoint16';
perfAblations(1,:) = mainViewpoint(classInds, usePascalViews); %% prints accuracy for bus, motorbike, dog, sheep respectively

params.features = 'vggCommon16';
perfAblations(2,:) = mainViewpoint(classInds, usePascalViews); %% prints accuracy for bus, motorbike, dog, sheep respectively

%% Run experiment for various similarity features

params.similarityFeatName =  'vggConv5';
for c = 1:nClasses
    [~,optAcc,optAccConf] = optimizePredictions(pascalIndexClass(classInds(c),'pascal'),0,usePascalViews(c),1,0,0);
    perfAblations(3,c) = optAcc;
    perfConfAblations(1,c) = optAccConf;
end

params.spatialNormSmoothing = 0; %no normalization
for c = 1:nClasses
    [~,optAcc,optAccConf] = optimizePredictions(pascalIndexClass(classInds(c),'pascal'),0,usePascalViews(c),1,0,0);
    perfAblations(4,c) = optAcc;
    perfConfAblations(2,c) = optAccConf;
end

params.similarityFeatName =  'vggFc7';
for c = 1:nClasses
    [~,optAcc,optAccConf] = optimizePredictions(pascalIndexClass(classInds(c),'pascal'),0,usePascalViews(c),1,0,0);
    perfAblations(5,c) = optAcc;
    perfConfAblations(3,c) = optAccConf;
end

end