function params = getParams()
%GET_PARAMS Summary of this function goes here
%   Detailed explanation goes here

params.vpsDataset = 'Joint';
params.features = 'vggCommon16'; %see cache/features folder to see what features are allowed
params.angleEncoding = 'euler'; %'euler' or 'rot' or 'cos_sin' or 'axisAngle'. 'cos_sin' is highly recommended !
params.optMethod = 'bin'; %'bin' or 'svr' or 'lsq' // don't use 'rf' for now as it breaks down
params.candidateThresh = 0.5; %IoU threshold for candidates to be used to in training. Dont set less than 0.5 as it might assigne same candidate to multiple gts
params.nHypotheses = 3;

params.classInds = [1 2 4 5 6 7 9 11 14 18 19 20]; %rigid categories
params.articulatedInds = [3 8 10 12 13 15 16 17]; %no sheep and dog
params.ilsvrcAnimals = [4 6 20 34 35 39 59 60 64 70 72 74 75 84 90 92 96 ...
    97 99 103 106 120 134 141 144 152 157 158 166 198 200];

params.ilsvrcRigid = [8 38 40 42 77 80 81 83 94 161 173 175 176 178 183 191 192];

params.excludeOccluded = false;
params.filterBoxes = true; %set to true if you want to weed out small boxes

params.featSimilarityThresh = 0.05;
params.rotSimilarityThresh = 30; %degrees
params.similarityFeatName  = 'vggConv5';
params.spatialNormSmoothing = 1;

end