function [accuracy,isGood,isCorrect] = evaluatePascalViews(azimuths,views)
%EVALUATEP Summary of this function goes here
%   Detailed explanation goes here

N = numel(views);Ngood = 0;
correct = 0;
for i=1:N
    isGood(i)=0;isCorrect(i) = 0;
    if(~isempty(views{i}))
        isGood(i) = 1;
        Ngood = Ngood+1;
        if(strcmp(azToView(azimuths(i)),views{i}))
            isCorrect(i) = 1;
            correct = correct+1;
        end
    end
end
accuracy = correct/Ngood;
end