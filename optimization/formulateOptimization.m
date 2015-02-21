function [initPreds,similarFeatInds,similarRotIndLabels] = formulateOptimization(preds,feat,encoding,predUnaries,softAssignment)
%FORMULATEOPTIMIZATION Summary of this function goes here
%   Detailed explanation goes here
% initPreds is an array of struct withs (choice,rots,unary)
% similarFeatInds{i} lists inds of instances with similar feat
% similarRotIndLabels{i,c} is a struct with instanceInds and choiceIndex

%% init
globals;
C = length(preds); %number of choices
N = size(preds{1},1);

%% rotations
for i=1:C
    eulersPred{i} = decodePose(preds{i},encoding);
    rotsPred{i} = encodePose(eulersPred{i},'rot');
end

rotsAll = {};

feat = (cell2mat(feat));
for n=1:N
    for c = 1:C
        rotsAll{n,c} = dcm2quat(reshape(rotsPred{c}(n,:),3,3));
    end
end

%% initpreds
initPreds = struct('unary',{},'rots',{},'probs',{},'choice',{});

for n = 1:N
    predThis = struct;
    predThis.unary = predUnaries(n,:);
    predThis.rots = {};
    if(softAssignment)
        predThis.probs = predThis.unary - log(sum(exp(predThis.unary)));
    else
        predThis.probs = -Inf(size(predThis.unary));
        [~,maxInd] = max(predThis.unary);
        predThis.probs(maxInd) = 0;
    end
    
    for c = 1:C
        predThis.rots{c} = rotsAll{n,c};
    end
    predThis.choice = 1;
    initPreds(n) = predThis;
end

rotsAllQ = {};
for c=1:C
    rotsAllQ{c} = cell2mat(rotsAll(:,c));
end    

%% similarfeatInds
similarFeatInds = {};
disp('Similar feat computing');
for n=1:N
    if(mod(n,50)==0)
        disp([num2str(n) '/' num2str(N)]);
    end
    featDists = sum(abs(bsxfun(@minus,feat,feat(n,:))),2);
    %featDists = cellfun(@(x) featDistErr(x,feat{n}), feat);
    featDists(n) = Inf;
    [~,idx] = sort(featDists,'ascend');
    similarFeatInds{n} = idx(1:round(N*params.featSimilarityThresh));
end

%% similarRotIndLabels
similarRotIndLabels = {};
disp('Similar rots computing');

for n=1:N
    if(mod(n,100)==0)
        disp([num2str(n) '/' num2str(N)]);
    end
    for c=1:C
        similarRotInds = struct;
        %rotDists = cellfun(@(r) relativeRotationMagnitude(rotsAll{n,c},r),rotsAll);
        
        rotDists = zeros(N,C);
        q1 = quatinv(rotsAll{n,c});
        q1(2:4)=-q1(2:4);
        for c2 = 1:C
            angles = acos(rotsAllQ{c2}*q1');
            rotDists(:,c2) = min([angles,pi-angles],[],2)*360/pi;
        end
        
        rotDists(n,:) = Inf;
        rotDists = (rotDists <= params.rotSimilarityThresh);
        [similarRotInds.instanceInds,similarRotInds.choiceIndex] = ind2sub(size(rotDists),find(rotDists));
        similarRotIndLabels{n,c} = similarRotInds;
    end
end


end


function err = relativeRotationMagnitude(q1,q2)
    q = quatdivide(q1,q2);
    angle = acos(q(1));
    err = min(angle,pi-angle)*360/pi;
    % R1 = quat2dcm(q1);R2 = quat2dcm(q2);
    % err2 = norm(logm(R1'*R2),'fro')/sqrt(2)*180/pi;
    % disp(err-err2)    
end

function err = featDistErr(f1,f2)
    err = sum(abs((f1)-(f2)));
end