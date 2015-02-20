function [preds] = optimizePredictions(class,useSaved)
%OPTIMIZEPREDICTIONS Summary of this function goes here
%   Detailed explanation goes here
globals;
encoding = params.angleEncoding;
goodInds = createEvalSets(class,0); %no train/val/test splits

%% Predictions from CNN
data = load(fullfile(cachedir,'evalSets',class));
[~,~,testLabels,~,~,testFeats] = generateEvalSetData(data);
[testPreds,predUnaries] = poseHypotheses(testFeats,params.nHypotheses,0);

%% set-up
formulationDir = fullfile(cachedir,['optimizationInit' params.vpsDataset],params.features);
mkdirOptional(formulationDir);
if(useSaved && exist(fullfile(formulationDir,[class '.mat']),'file'))
    load(fullfile(formulationDir,[class '.mat']))
else
    load(fullfile(cachedir,['rcnnPredsVps' params.vpsDataset],'vggConv5',class));
    feat = feat(goodInds);
    
    for i=1:length(feat)
        tmp = reshape(sigmoid(feat{i}),[14 14 512]);
        for c = 1:512
            tmp(:,:,c)=tmp(:,:,c)/sum(sum(tmp(:,:,c)));
        end
        feat{i} = (tmp(:))';
    end
        
    [initPreds,similarFeatInds,similarRotIndLabels] = formulateOptimization(testPreds,feat, encoding,predUnaries);
    save(fullfile(formulationDir,[class '.mat']),'initPreds','similarFeatInds','similarRotIndLabels','feat');
end
keyboard;

currentPreds = initPreds;
preds = [];
for n=1:length(initPreds)
    preds(n,:) = testPreds{currentPreds(n).choice}(n,:);
end

%% iterate

for iter = 1:50
    if(iter > 1 && numFlips <= 2)
        continue;
    end
    numFlips = 0;
    disp(iter);
    for n=randperm(length(initPreds))
        [choiceIndex,choiceScores] = updateRotationChoice(n,currentPreds,similarFeatInds,similarRotIndLabels,feat);

%         if(isinf(min(choiceScores)))
%             visFeatNeighbors(data.test,similarFeatInds{n},n);
%             pause();close all;
%         end
%             
        if(choiceIndex ~= currentPreds(n).choice)
            %scores = choiceScores([currentPreds(n).choice choiceIndex]);
            %prediction = [testPreds{currentPreds(n).choice}(n,:);testPreds{choiceIndex}(n,:)];
            %visualizeOptimizationSwitch(prediction,encoding,class,data.test.voc_ids{n},data.test.dataset{n},data.test.bboxes(n,:),data.test.objectInds(n),scores);
            if(choiceScores(choiceIndex) - choiceScores(currentPreds(n).choice) > 1/iter)
                currentPreds(n).choice = choiceIndex;
                preds(n,:) = testPreds{currentPreds(n).choice}(n,:);
                numFlips = numFlips+1;
            end
        end
    end
    [testErrsOpt] = evaluatePredictionError({preds},testLabels,encoding,0);
    testAccuracyOpt = sum(testErrsOpt<=30)/numel(testErrsOpt)
    testMedErrorOpt = median(testErrsOpt)
    disp(numFlips);
end

%% visualization and results
% preds = [];
% for n=1:length(initPreds)
%    preds(n,:) = testPreds{currentPreds(n).choice}(n,:);
%    [choiceIndex,choiceScores] = updateRotationChoice(n,currentPreds,similarFeatInds,similarRotIndLabels,feat);
%     if(choiceIndex ~= 1)
%         scores = choiceScores([initPreds(n).choice choiceIndex]);
%         prediction = [testPreds{initPreds(n).choice}(n,:);testPreds{choiceIndex}(n,:)];
%         visualizeOptimizationSwitch(prediction,encoding,class,data.test.voc_ids{n},data.test.dataset{n},data.test.bboxes(n,:),data.test.objectInds(n),scores);
%         figure();
%         visFeatNeighbors(data.test,similarFeatInds{n},n);
%         pause();close all;
%     end
% end

%% eval
[testErrsOpt] = evaluatePredictionError({preds},testLabels,encoding,0);
testAccuracyOpt = sum(testErrsOpt<=30)/numel(testErrsOpt);
testMedErrorOpt = median(testErrsOpt);

[testErrs] = evaluatePredictionError(testPreds,testLabels,encoding,0);
testAccuracy = sum(testErrs<=30)/numel(testErrs);
testMedError = median(testErrs);

keyboard;


end

