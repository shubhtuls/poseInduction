function [choiceIndex,choiceScores] = updateRotationChoice(variableIndex,currentPreds,similarFeatInds)

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

    if(numSimilarRots~=0)
        consistentFeatScore = log(sum(consistentFeatInds)) - 0.5*log(numSimilarRots);
    else
        consistentFeatScore = 0;
    end

    scoreThis = unaryScore + consistentFeatScore;
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

threshRatio = 2/3;

q1 = quatinv(rot);
q1(2:4)=-q1(2:4);
rots = vertcat(rotPreds(:).rots);
closeInds = zeros(size(rots,1),1);
probs = exp(vertcat(rotPreds(:).probs));

for c=1:size(rots,2)
    rotsC = cell2mat(rots(:,c));

    angles = real(acos(rotsC*q1'));
    rotDists = min([angles,pi-angles],[],2)*360/pi;

    closeInds = closeInds + probs(:,c).*(threshRatio*double(rotDists <= thresh) + (1-threshRatio)*double(rotDists <= 2*thresh));
end

end

function err = featDistErr(f1,f2)
    err = sum(abs((f1)-(f2)));
end