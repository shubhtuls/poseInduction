function [] = mainPredict()
%MAINPREDICT Summary of this function goes here
%   Detailed explanation goes here

%% predictions for SCT
generatePoseFeatures('vggJoint16','vggJoint16',224,[6 12 14 17],0,0,[7 9 2 10]);
caffe.reset_all();

%% predictions for GC
generatePoseFeatures('vggCommon16','vggCommon16',224,[6 12 14 17],0,0);
caffe.reset_all();

%% add VGG conv5 feature computation here


end