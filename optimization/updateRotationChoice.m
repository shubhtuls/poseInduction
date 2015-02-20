function [choiceIndex,choiceScores] = updateRotationChoice(variableIndex,currentPreds,similarFeatInds,similarRotIndLabels,featAll)

globals;
score = -Inf;
C = length(currentPreds(variableIndex).rots);
currentChoices = [currentPreds(:).choice];
choiceScores = zeros(1,C);
choiceIndex = 1;

for c = 1:C
    rot = currentPreds(variableIndex).rots{c};
    unaryScore = currentPreds(variableIndex).unary(c);

    similarFeatPreds = currentPreds(similarFeatInds{variableIndex});
    consistentFeatInds = similarRots(rot,similarFeatPreds,params.rotSimilarityThresh);
    numSimilarRots = sum(similarRots(rot,currentPreds,params.rotSimilarityThresh));
    
    % disp(sum(similarRots(rot,currentPreds,params.rotSimilarityThresh)));
    
    %if(sum(consistentFeatInds)==0)
        %keyboard;
        %close all;
    %end
    if(numSimilarRots~=0)
        consistentFeatScore = log(sum(consistentFeatInds)) - 0.5*log(numSimilarRots);
    else
        consistentFeatScore = 0;
    end
%     similarRotInds = similarRotIndLabels{c}.instanceInds;
%     similarRotAssignment = similarRotIndLabels{c}.choiceIndex;
%     goodInds = (similarRotAssignment == currentChoices(similarRotInds));
%     similarRotInds = similarRotInds(goodInds);
%     similarRotFeat = featAll(similarRotInds);
%     featThis = featAll{variableIndex};
%     consistentRotDist = cellfun( @(x) featDistErr(featThis,x),similarRotFeat);
%     consistentRotScore = sum(consistentRotDist < params.featSimilarityThresh)/numel(consistentRotDist);
    consistentRotScore = 1;
    scoreThis = unaryScore + consistentFeatScore + log(consistentRotScore);
    choiceScores(c) = scoreThis;
    if(scoreThis > score)
        score = scoreThis;
        choiceIndex = c;
    end
end

end

function err = relativeRotationMagnitude(q1,q2)
    q = quatdivide(q1,q2);
    angle = acos(q(1));
    err = min(angle,pi-angle)*360/pi;
end

function closeInds = similarRots(rot,rotPreds,thresh)

%closeInds = false(length(rotPreds),1);

q1 = quatinv(rot);
q1(2:4)=-q1(2:4);
rots = vertcat(rotPreds(:).rots);
cInds = (vertcat(rotPreds(:).choice));
rots = cell2mat(rots(sub2ind(size(rots),1:size(rots,1),cInds'))');

angles = real(acos(rots*q1'));
rotDists = min([angles,pi-angles],[],2)*360/pi;

closeInds = 2/3*double(rotDists <= thresh) + 1/3*double(rotDists <= 2*thresh);

end

function err = featDistErr(f1,f2)
    err = sum(abs((f1)-(f2)));
end