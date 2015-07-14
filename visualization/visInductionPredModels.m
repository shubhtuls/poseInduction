function [] = visInductionPredModels(class)
%VISINDUCTIONPREDMODELS Summary of this function goes here
%   Detailed explanation goes here

globals;
encoding = params.angleEncoding;
var = load(fullfile(cachedir,'inductionPreds',class));
data.test = var.inductionData;
[~,~,testLabels,~,~,testFeats] = generateEvalSetData(data);
%keyboard;
[testErrsOpt] = evaluatePredictionError({var.preds},testLabels,encoding,0);
saveVisualizations(class,var.preds,data.test,encoding,testErrsOpt,'ascend');

end