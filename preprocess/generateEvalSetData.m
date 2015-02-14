function [trainLabels,valLabels,testLabels,trainFeats,valFeats,testFeats] = generateEvalSetData(data)
%GENERATEEVALSETLABELS Summary of this function goes here
%   Detailed explanation goes here

globals;
encoding = params.angleEncoding;

trainLabels = [];valLabels=[];testLabels=[];
trainFeats = [];valFeats = [];testFeats = [];

if(isfield(data,'train'))
    trainLabels = encodePose(data.train.eulers,encoding);
    trainFeats = double(data.train.feat);
end

if(isfield(data,'val'))
    valLabels = encodePose(data.val.eulers,encoding);
    valFeats = double(data.val.feat);
end

if(isfield(data,'test'))
    testLabels = encodePose(data.test.eulers,encoding);
    testFeats = double(data.test.feat);
end

end