function [batches, batch_padding] = cnnExtractRegionsSingleBox(dataStruct, cnn_model,mirror)

globals;
imgLoc = 0;

if(isfield(dataStruct,'imgDir') && isfield(dataStruct,'imgExt'))
    imgLoc = 1;
end

singleImage = length(dataStruct.voc_image_id) == 1;
if(singleImage)
    if(imgLoc)
        imgDir = dataStruct.imgDir{1};
        imgExt = dataStruct.imgExt{1};
    else
        [imgDir,imgExt] = getDatasetImgDir(dataStruct.dataset{1});
    end
    im = imread(fullfile(imgDir,[dataStruct.voc_image_id{1} imgExt]));
    if(size(im,3)==1)
        im(:,:,2) = im(:,:,1);
        im(:,:,3) = im(:,:,1);
    end

    im = single(im(:,:,[3 2 1]));
end

num_boxes = size(dataStruct.bbox,1);
batch_size = cnn_model.batch_size;
num_batches = ceil(num_boxes / batch_size);
batch_padding = batch_size - mod(num_boxes-1, batch_size) - 1;

crop_mode = cnn_model.crop_mode;
image_mean = cnn_model.image_mean;
crop_size = size(image_mean,1);
crop_padding = cnn_model.crop_padding;

batches = cell(num_batches, 1);
for batch = 1:num_batches
%  disp(batch);
%parfor batch = 1:num_batches
  batch_start = (batch-1)*batch_size+1;
  batch_end = min(num_boxes, batch_start+batch_size-1);

  ims = zeros(crop_size, crop_size, 3, batch_size, 'single');
  for j = batch_start:batch_end
    bbox = dataStruct.bbox(j,:);
    if(~singleImage)
        if(imgLoc)
            imgDir = dataStruct.imgDir{j};
            imgExt = dataStruct.imgExt{j};
        else
            [imgDir,imgExt] = getDatasetImgDir(dataStruct.dataset{j});
        end
        im = imread(fullfile(imgDir,[dataStruct.voc_image_id{j} imgExt]));
        if(size(im,3)==1)
            im(:,:,2) = im(:,:,1);
            im(:,:,3) = im(:,:,1);
        end
        im = single(im(:,:,[3 2 1]));
    end

    [crop] = cnn_im_crop(im, bbox, crop_mode, crop_size, ...
        crop_padding, image_mean);
    if(mirror)
        %size(crop)
        for k=1:size(crop,3)
            crop(:,:,k) = fliplr(crop(:,:,k));
        end
    end
    % swap dims 1 and 2 to make width the fastest dimension (for caffe)
    ims(:,:,:,j-batch_start+1) = permute(crop, [2 1 3]);
  end

  batches{batch} = ims;
end
