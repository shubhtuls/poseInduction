function [] = visInductionPredClusters(class)
%VISINDUCTIONPREDMODELS Summary of this function goes here
%   Detailed explanation goes here

globals;
encoding = params.angleEncoding;
var = load(fullfile(cachedir,'inductionPreds',class));
data.test = var.inductionData;
%[~,~,testLabels,~,~,testFeats] = generateEvalSetData(data);
%keyboard;
%[testErrsOpt] = evaluatePredictionError({var.preds},testLabels,encoding,0);
%saveVisualizations(class,var.preds,data.test,encoding,testErrsOpt,'ascend');

visLabels = var.preds(:,[2 3]);
visLabels(:,2) = mod(visLabels(:,2) + pi,2*pi)-pi;
[~,topInds] = sort(var.scores,'descend');
%topInds = topInds(1:round(numel(topInds)/3));

imSaveDir = fullfile(cachedir,'images','poseClustersOpt',class);mkdirOptional(imSaveDir);
delete([imSaveDir '/*']);
visPoseClusters(visLabels(:,2), data.test.voc_ids, data.test.dataset, data.test.bboxes,topInds, imSaveDir);

end