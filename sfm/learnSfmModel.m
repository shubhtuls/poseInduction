function [model] = learnSfmModel(dataStruct,goodInds)
%LEARNSFMMODEL Summary of this function goes here
%   Detailed explanation goes here

if(iscell(goodInds)) %passed the fnames
    goodInds = ismember(dataStruct.voc_image_id,goodInds);
end
goodInds = goodInds' & cellfun(@(x) ~isempty(x) && (sum(~isnan(x(:)))>=6),dataStruct.kps); %atleast 3 visible keypoints


D = size(dataStruct.kps{1},1);

for i=1:length(dataStruct)
    bbox = dataStruct.bbox(i,:);
    maxDim = max(bbox([4 3]) - bbox([2 1]));
    dataStruct.kps{i} = (dataStruct.kps{i} - repmat(bbox([3 4]) + bbox([1 2]),D,1)/2)*10/maxDim;
end

P = (horzcat(dataStruct.kps{goodInds}));P = P';
max_em_iter = 100;

%keyboard;

[~, S, ~] = sfmFactorization(P, 200, 10);

%keyboard;
model.S = S;
%model.P3s = P3; %just so we sample intelligently
%model.cs = c;
%model.Trs = Tr;
%model.defBasis = [];
%model.part_names = dataStruct(1).part_names;
%visualize_3D_basis_shapes(model,wireframe_car());

end

