function [testErrors,testMedErrors,testErrs,testData,testPreds,testLabels] = regressToPose(class, evalPascalViews)
%[testErrors,testMedErrors] = regressToPose(class)
%   uses the training/val/test sets specified in parameters and
% regresses to pose and returns error

%% Initializations
%disp(class);
globals;
%params = getParams();
encoding = params.angleEncoding;
createEvalSets(class,0);

%% Loading Data

%data = load(fullfile(cachedir,'splitSets',class));
data = load(fullfile(cachedir,'evalSets',class));
[~,~,testLabels,~,~,testFeats] = generateEvalSetData(data);

%% TESTING
switch params.optMethod        
    case 'bin'
        alphaOpt = 0;
        nHypotheses = params.nHypotheses;
        %disp(alphaOpt)
        %[trainPreds] = poseHypotheses(trainFeats,nHypotheses,alphaOpt);
        %trainPred = trainPreds{1};
        
        [testPreds] = poseHypotheses(testFeats,nHypotheses,alphaOpt);
        testPreds = testPreds{1};
        %[testPreds] = bestPoseKeypointCandidate(testPreds,data.test,class);
        
        %[valPreds,valSubtypes] = poseHypotheses(valFeats,nHypotheses,alphaOpt);
        %valPreds = valPreds{1};
        %[valPreds] = bestPoseKeypointCandidate(valPreds,data.val,class);
end

%keyboard;
testErrs = evaluatePredictionError(testPreds,testLabels,encoding,0);
if(evalPascalViews)
    [accPascalViews] = evaluatePascalViews(testPreds(:,3),data.test.views);
end
%[valErrs,bestValPred] = evaluatePredictionError(valPreds,valLabels,encoding,0);
%[trainErrs,bestTrainPred] = evaluatePredictionError(trainPred,trainLabels,encoding,0);

%diff = testPreds - testLabels;
%mean(sum(diff.*diff,2));
testErrors = [];testMedErrors=[];
%testErrs = [valErrs;testErrs];

% plot(sort(valErrs),[1:length(valErrs)]./length(valErrs),'r');hold on;
% plot([0 180],[0 1],'k');
% grid on;xlim([0 180]);ylim([0 1]);
% pause();close all;

%testErrs = [valErrs;testErrs]; %% using all objects

testErrors(1) = sum(testErrs<=30)/numel(testErrs);
testMedErrors(1) = median(testErrs);
if(evalPascalViews)
    [accPascalViews] = evaluatePascalViews(testPreds(:,3),data.test.views);
    testErrors(1) = accPascalViews;
end
testData = data.test;

%[errSort,IDX] = sort(testErrs,'ascend');
%plot(errSort,[1:length(errSort)]/length(errSort));pause();close all;
%visualizePredictions(class,testPreds{1},data.test,encoding,testErrs,'image');
%visualizePredictions(class,testPreds,data.test,encoding,testErrs,'image');
%saveVisualizations(class,valPreds{1},data.val,encoding,valErrs,'ascend');

end
