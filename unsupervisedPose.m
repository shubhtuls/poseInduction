function [testAccuracy,testMedError,testErrs,testData,testPreds,testLabels] = unsupervisedPose(class)
%[testErrors,testMedErrors] = regressToPose(class)
%   uses the training/val/test sets specified in parameters and
% regresses to pose and returns error

%% Initializations
%disp(class);
globals;
%params = getParams();
encoding = params.angleEncoding;
createEvalSets(class,0); %no train/val/test splits

%% Loading Data

data = load(fullfile(cachedir,'evalSets',class));
[~,~,testLabels,~,~,testFeats] = generateEvalSetData(data);

%% TESTING
switch params.optMethod        
    case 'bin'
        alphaOpt = 0;
        nHypotheses = params.nHypotheses;
        
        [testPreds] = poseHypotheses(testFeats,nHypotheses,alphaOpt);
        %testPreds = testPreds{1};
        %[testPreds] = bestPoseKeypointCandidate(testPreds,data.test,class);
end

%keyboard;
[testErrs,bestTestPred] = evaluatePredictionError(testPreds,testLabels,encoding,0);
testAccuracy = sum(testErrs<=30)/numel(testErrs);
testMedError = median(testErrs);
testData = data.test;

%[errSort,IDX] = sort(testErrs,'ascend');
%plot(errSort,[1:length(errSort)]/length(errSort));
%pause();close all;

%visualizePredictions(class,testPreds{1},data.test,encoding,testErrs);
visualizePredictions(class,bestTestPred,data.test,encoding,testErrs);
%visualizePredictions(class,testPreds{1},data.test,encoding,testErrs,'ascend');

end
