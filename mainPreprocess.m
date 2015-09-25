function [] = mainPreprocess()
%MAINPREPROCESS Summary of this function goes here
%   Detailed explanation goes here

globals;
%% rigid classes
for c = params.classInds
    class = pascalIndexClass(c, 'pascal');
    readData(class);
end

%% articulated classes
for c = params.articulatedInds
    class = pascalIndexClass(c, 'pascal');
    pascalData = readDataPascal(class,'2012');
    fname = fullfile(rotationPascalDataDir,class);
    rotationData = pascalData;
    save(fname,'rotationData');
end

for c = params.articulatedInds
    class = pascalIndexClass(c, 'pascal');
    augmentArticulatedPose(class);
end

%% generate files for cnn training
vpsPascalDataCollect()
vpsImagenetDataCollect()
rcnnBinnedJointTrainValTestCreate('');

end
