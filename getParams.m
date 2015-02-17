function params = getParams()
%GET_PARAMS Summary of this function goes here
%   Detailed explanation goes here

params.vpsDataset = 'Joint';
params.features = 'vggCommon10'; %see cache/features folder to see what features are allowed
params.angleEncoding = 'euler'; %'euler' or 'rot' or 'cos_sin' or 'axisAngle'. 'cos_sin' is highly recommended !
params.optMethod = 'bin'; %'bin' or 'svr' or 'lsq' // don't use 'rf' for now as it breaks down
params.candidateThresh = 0.5; %IoU threshold for candidates to be used to in training. Dont set less than 0.5 as it might assigne same candidate to multiple gts
params.nHypotheses = 1;

params.classInds = [1 2 4 5 6 7 9 11 14 18 19 20]; %rigid categories

params.excludeOccluded = false;
params.filterBoxes = true; %set to true if you want to weed out small boxes

end