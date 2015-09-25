function [preds, testAccuracyOpt, testAccuracyOptConf] = optimizePredictions(class,useSaved,evalPascalViews,useMirror,useSoftAssignment,azimuthOnly)
%OPTIMIZEPREDICTIONS Summary of this function goes here
%   Detailed explanation goes here
% useSaved is 0/1 : 1 means it uses
% useMirror is 0/1 : 1 uses mirrored instances as well for optimization
% useSoftAssignment : use hard or soft assignments
% azimuthOnly : uses only azimuth similarity for optimization (works similar either way)

globals;
encoding = params.angleEncoding;
goodInds = createEvalSets(class,0); %no train/val/test splits
visSwitches = 0;
visErrors = 0;

%% Predictions from CNN
data = load(fullfile(cachedir,'evalSets',class));
[~,~,testLabels,~,~,testFeats] = generateEvalSetData(data);

visLabels = testLabels(:,[2 3]);
visLabels(:,2) = mod(visLabels(:,2) + pi,2*pi);
%showEmbedding(visLabels, data.test.voc_ids, data.test.dataset, data.test.bboxes);
%keyboard;
%close all;

[testPredsAec,predUnaries] = poseHypotheses(testFeats,params.nHypotheses,0);
N = size(testPredsAec{1},1);

testPreds = testPredsAec;
if(azimuthOnly)
    for i=1:length(testPredsAec)
        testPreds{i}(:,1:2)=0;
    end
end

%% mirrored preds
predUnaries = vertcat(predUnaries,predUnaries);
testPredsMirror = testPreds;
if(useMirror)
    for i=1:length(testPreds)
        testPredsMirror{i} = vertcat(testPreds{i},mirrorFlipPreds(testPreds{i}));
    end
end
predsQuat = zeros(N,4);
predsQuatMirror = zeros(N,4);
testLabelsMirror = vertcat(testLabels,mirrorFlipPreds(testLabels));
rotsLabelMirror = encodePose(testLabelsMirror,'rot');

rotX = diag([1 -1 -1]);
for n=1:N
    rotThis = rotX*(reshape(rotsLabelMirror(n,:),3,3))';
    predsQuat(n,:) = dcm2quat(rotThis);
    if(predsQuat(n,1) <0)
        predsQuat(n,:) = -predsQuat(n,:);
    end
    rotThis = rotX*(reshape(rotsLabelMirror(n+N,:),3,3))';
    predsQuatMirror(n,:) = dcm2quat(rotThis);
    if(predsQuatMirror(n,1) <0)
        predsQuatMirror(n,:) = -predsQuatMirror(n,:);
    end
end
%keyboard;
similarityFeatName = params.similarityFeatName; %'vggConv5';
spatialFeat = ~isempty(strfind(similarityFeatName,'Pool')) || ~isempty(strfind(similarityFeatName,'Conv'));
spatialNormSmoothing = params.spatialNormSmoothing;

%% set-up optimization
formulationDir = fullfile(cachedir,['optimizationInit' params.vpsDataset],params.features);
mkdirOptional(formulationDir);
if(useSaved && exist(fullfile(formulationDir,[class '.mat']),'file'))
    load(fullfile(formulationDir,[class '.mat']));
else
    fg = fspecial('gaussian',[3 3],1);
    var = load(fullfile(cachedir,['rcnnPredsVps' params.vpsDataset],similarityFeatName,class));
    %load(fullfile(cachedir,['rcnnPredsVps' params.vpsDataset],'vggSimilarityCommon16',class));
    feat = var.featStruct{1};feat = feat(goodInds);
    if(useMirror)
        featMirror = var.featStructMirror{1};
        featMirror = featMirror(goodInds);
        feat = vertcat(feat,featMirror);
    end
    if(spatialFeat && spatialNormSmoothing)
        for i=1:length(feat)
            tmp = reshape(sigmoid(feat{i}),[14 14 512]);
            for c = 1:512
                tmp(:,:,c)=tmp(:,:,c)/sum(sum(tmp(:,:,c)));
                %tmp(:,:,c) = conv2(tmp(:,:,c),fg,'same');
            end
            feat{i} = (tmp(:))';
        end
    else
        for i=1:length(feat)
            feat{i} = sigmoid(feat{i});
            feat{i} = (feat{i}(:))';
        end
    end
        
    [initPreds,similarFeatInds,similarRotIndLabels] = formulateOptimization(testPredsMirror,feat, encoding, predUnaries,useSoftAssignment);
    save(fullfile(formulationDir,[class '.mat']),'initPreds','similarFeatInds','similarRotIndLabels','feat');
end

for n=1:length(initPreds)
    if(useSoftAssignment)
        initPreds(n).probs = initPreds(n).unary - log(sum(exp(initPreds(n).unary)));
    else
        initPreds(n).probs = -Inf(size(initPreds(n).unary));
        [~,maxInd] = max(initPreds(n).unary);
        initPreds(n).probs(maxInd) = 0;
    end
end
%keyboard;

currentPreds = initPreds;
preds = [];scoreDiff = [];
for n=1:N
    preds(n,:) = testPredsAec{currentPreds(n).choice}(n,:);
    unarySort = sort(initPreds(n).unary,'descend');
    %scoreDiff(n) = unarySort(1)-unarySort(2);
    scoreDiff(n) = unarySort(1);
end

%% initial visualization
%visPoseManifold(preds,scoreDiff,data.test,[2:3:20],1.1);
%pause();close all;

%% iterate
for iter = 1:50
    if(iter > 1 && numFlips <= 2)
        continue;
    end
    numFlips = 0;
    disp(iter);
    for n=randperm(length(initPreds))
        [choiceIndex,choiceScores] = updateRotationChoice(n,currentPreds,similarFeatInds);
        if(useSoftAssignment)
            probs = choiceScores - log(sum(exp(choiceScores)));
        else
            probs = -Inf(size(currentPreds(n).probs));
            probs(choiceIndex) = 0;
        end
        if(n<=N)
            unarySort = sort(choiceScores,'descend');
            %scoreDiff(n) = unarySort(1)-unarySort(2);
            scoreDiff(n) = unarySort(1);
        end
        currentPreds(n).probs = probs;
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
            %if(choiceScores(choiceIndex) - choiceScores(currentPreds(n).choice) > 0)
                currentPreds(n).choice = choiceIndex;
                if(n<=N)
                    preds(n,:) = testPredsAec{currentPreds(n).choice}(n,:);
                end
                numFlips = numFlips+1;
            end
        end
    end
    
    [testErrsOpt] = evaluatePredictionError({preds},testLabels,encoding,0);
    testAccuracyOpt = sum(testErrsOpt<=30)/numel(testErrsOpt);
    testMedErrorOpt = median(testErrsOpt);
    disp(numFlips);
end

%% confident preds
for n=1:N
    [~,choiceScores] = updateRotationChoice(n,currentPreds,similarFeatInds);
    unarySort = sort(choiceScores,'descend');
    scores(n) = unarySort(1);
    scoreDiffs(n) = unarySort(1) - unarySort(2);
end
[~,topInds] = sort(scores,'descend');
topInds = topInds(1:round(N/3));


%% eval
[testErrsOpt] = evaluatePredictionError({preds},testLabels,encoding,0);
[testErrsOptConf] = evaluatePredictionError({preds(topInds,:)},testLabels(topInds,:),encoding,0);

testAccuracyOpt = sum(testErrsOpt<=30)/numel(testErrsOpt);
testAccuracyOptConf = sum(testErrsOptConf<=30)/numel(testErrsOptConf);

testMedErrorOpt = median(testErrsOpt);

[testErrsBase] = evaluatePredictionError(testPredsAec(1),testLabels,encoding,0);
testAccuracyBase = sum(testErrsBase<=30)/numel(testErrsBase);
testMedErrorBase = median(testErrsBase);

[testErrs,predBest] = evaluatePredictionError(testPredsAec,testLabels,encoding,0);
testAccuracy = sum(testErrs<=30)/numel(testErrs);
testMedError = median(testErrs);

%% eval left/right/frontal
if(evalPascalViews)
    [accLabels,~,isCorrectLabels] = evaluatePascalViews(testLabels(:,3),data.test.views);
    %accLabels
    [accOpt,isGoodOpt,isCorrectOpt] = evaluatePascalViews(preds(:,3),data.test.views);
    [accOptConf,~,~] = evaluatePascalViews(preds(topInds,3),data.test.views(topInds));
    testAccuracyOpt = accOpt;
    testAccuracyOptConf = accOptConf;
    evaluatePascalViews(testPredsAec{1}(:,3),data.test.views)
end

%% visualize pose manifold
%visPoseManifold(preds,scoreDiff,data.test,[2:3:20],1);
%pause();close all;

%% visualization - flipped instances
if(visSwitches)
    preds = [];
    for n=1:N
       preds(n,:) = testPreds{currentPreds(n).choice}(n,:);
       [choiceIndex,choiceScores] = updateRotationChoice(n,currentPreds,similarFeatInds);
        if(choiceIndex ~= 1)
            scores = choiceScores([initPreds(n).choice choiceIndex]);
            prediction = [testPreds{initPreds(n).choice}(n,:);testPreds{choiceIndex}(n,:)];
            visualizeOptimizationSwitch(prediction,encoding,class,data.test.voc_ids{n},data.test.dataset{n},data.test.bboxes(n,:),data.test.objectInds(n),scores);
            figure();
            visFeatNeighbors(data.test,similarFeatInds{n},n);
            pause();close all;
        end
    end
end

%% visualization - incorrect instances
if(visErrors)
    preds = [];
    for n=randperm(N)
       preds(n,:) = testPredsAec{currentPreds(n).choice}(n,:);
       [choiceIndex,choiceScores] = updateRotationChoice(n,currentPreds,similarFeatInds);
        if(testErrs(n) < testErrsOpt(n))
            for c = 1:length(testPredsAec)
                if(sum(abs(testPredsAec{c}(n,:)-predBest(n,:)))==0)
                    bestInd = c;
                end
            end
            scores = choiceScores([choiceIndex bestInd]);
            prediction = [preds(n,:);predBest(n,:)];
            visualizeOptimizationSwitch(prediction,encoding,class,data.test.voc_ids{n},data.test.dataset{n},data.test.bboxes(n,:),data.test.objectInds(n),scores);
            figure();
            visFeatNeighbors(data.test,similarFeatInds{n},n);
            pause();close all;
        end
    end
end

%% debug
visLabels = preds(:,[2 3]);
visLabels(:,2) = mod(visLabels(:,2) + pi,2*pi)-pi;
%showEmbedding(visLabels, data.test.voc_ids, data.test.dataset, data.test.bboxes, testErrs<=40)

%showEmbedding(visLabels, data.test.voc_ids, data.test.dataset, data.test.bboxes);
%visPoseClusters(visLabels(:,2), data.test.voc_ids, data.test.dataset, data.test.bboxes);

%showEmbedding(visLabels(topInds,:), data.test.voc_ids(topInds), data.test.dataset(topInds), data.test.bboxes(topInds,:));
%topInds = topInds(1:round(N/3));

%imSaveDir = fullfile(cachedir,'images','poseClustersOpt',class);mkdirOptional(imSaveDir);
%delete([imSaveDir '/*']);
%visPoseClusters(visLabels(:,2), data.test.voc_ids, data.test.dataset, data.test.bboxes,imSaveDir);
%visPoseClusters(visLabels(topInds,2), data.test.voc_ids(topInds), data.test.dataset(topInds), data.test.bboxes(topInds,:),imSaveDir);
%keyboard;

%% save

inductionData = data.test;
inductionDir = fullfile(cachedir,'inductionPreds');
mkdirOptional(inductionDir)
save(fullfile(inductionDir,class),'inductionData','preds','scores','scoreDiffs')

%keyboard;

end

function preds = mirrorFlipPreds(preds)
    preds(:,1) = - preds(:,1);
    preds(:,3) = 2*pi-preds(:,3);
end