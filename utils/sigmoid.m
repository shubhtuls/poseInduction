function [X] = sigmoid(X)
%SIGMOID Summary of this function goes here
%   Detailed explanation goes here

X = 1./(1+exp(-X));

end

