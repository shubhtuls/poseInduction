function [] = visClassNeighbors(class,featName)
%VISCLASSNEIGHBORS Summary of this function goes here
%   Detailed explanation goes here

globals;
goodInds = createEvalSets(class,0); %no train/val/test splits
data = load(fullfile(cachedir,'evalSets',class));
data = data.test;
load(fullfile(cachedir,['rcnnPredsVps' params.vpsDataset],featName,class));
feat = feat(goodInds);

%%
N = length(data.voc_ids);
if(numel(size(feat{1})) > 2)
    for i=1:length(feat)
        tmp = reshape(sigmoid(feat{i}),[14 14 512]);
        for c = 1:512
            tmp(:,:,c)=tmp(:,:,c)/sum(sum(tmp(:,:,c)));
            tmp(:,:,c) = conv2(tmp(:,:,c),fg,'same');
        end
        feat{i} = (tmp(:))';
    end
end
feat = (cell2mat(feat));
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

%% vis
for i = randperm(N,50)
    visFeatNeighbors(data,similarFeatInds{i},i);
    pause();close all;
end


end