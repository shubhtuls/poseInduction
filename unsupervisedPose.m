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
        
        [testPreds,predUnaries] = poseHypotheses(testFeats,nHypotheses,alphaOpt);
        %testPreds = testPreds{1};
        %[testPreds] = bestPoseKeypointCandidate(testPreds,data.test,class);
end

%keyboard;
[testErrs,bestTestPred] = evaluatePredictionError(testPreds(1),testLabels,encoding,0);
testAccuracy = sum(testErrs<=30)/numel(testErrs);
testMedError = median(testErrs);
testData = data.test;

evalPascalViews = 1;
if(evalPascalViews)
    [accLabels,~,isCorrectLabels] = evaluatePascalViews(testLabels(:,3),data.test.views);
    accLabels
    [accOpt,isGoodOpt,isCorrectOpt] = evaluatePascalViews(testPreds{1}(:,3),data.test.views);
    accOpt    
end

%[errSort,IDX] = sort(testErrs,'ascend');
%plot(errSort,[1:length(errSort)]/length(errSort));
%pause();close all;

%visualizePredictions(class,testPreds{1},data.test,encoding,testErrs);
%visualizePredictions(class,bestTestPred,data.test,encoding,testErrs);
%visualizePredictions(class,testPreds{1},data.test,encoding,testErrs,'ascend');


[~,topInds] = sort(predUnaries(:,1),'descend');
N = numel(topInds);
topInds = topInds(1:round(N/3));

visLabels = testPreds{1}(:,[2 3]);
visLabels(:,2) = mod(visLabels(:,2) + pi,2*pi)-pi;
%showEmbedding(visLabels, data.test.voc_ids, data.test.dataset, data.test.bboxes, testErrs<=40)
%showEmbedding(visLabels, data.test.voc_ids, data.test.dataset, data.test.bboxes)


imSaveDir = fullfile(cachedir,'images','poseClusters',class);mkdirOptional(imSaveDir);
delete([imSaveDir '/*']);
%visPoseClusters(visLabels(:,2), data.test.voc_ids, data.test.dataset, data.test.bboxes,imSaveDir);
visPoseClusters(visLabels(topInds,2), data.test.voc_ids(topInds), data.test.dataset(topInds), data.test.bboxes(topInds,:),imSaveDir);

end