function featStruct = cnnFeaturesSingleBox(dataStruct, cnn_model,mirror,labelsIn)
%% dataStruct has voc_image_id, bbox as fields
% make sure that caffe has been initialized for this model

if(nargin<3)
    mirror=0;
end
if(nargin<4)
    labelsIn = false;
end
% Each batch contains 256 (default) image regions.
% Processing more than this many at once takes too much memory
% for a typical high-end GPU.
disp('Extracting image patches');
[batches, batch_padding] = cnnExtractRegionsSingleBox(dataStruct, cnn_model, mirror);
batch_size = cnn_model.batch_size;
%disp(['numBatches = ' num2str(length(batches))])
%keyboard;
% compute features for each batch of region images

featStruct = {};
curr = 1;
disp('Computing Features');

for j = 1:length(batches)
  % forward propagate batch of region images
  %keyboard;
  if(labelsIn)
      labels = zeros(1,batch_size);
      labels(1:min(length(dataStruct.labels)-(j-1)*batch_size,batch_size)) = dataStruct.labels(curr:min(j*batch_size,length(dataStruct.labels)));
      f = cnn_model.net.forward({batches{j};single(labels)});
  else
      f = cnn_model.net.forward({batches{j};single(labels)});
  end
  for o = 1:length(cnn_model.net.outputs)
  	f{o} = permute(f{o},ndims(f{o})+1 - [1:ndims(f{o})]);
  end

  % keyboard;
  % first batch, init feat_dim and feat
  if j == 1
  	for o = 1:length(cnn_model.net.outputs)
	    feat_dim = size(f{o});
	    feat_dim(1) = size(dataStruct.bbox,1);
    	%featStruct{o} = zeros(feat_dim, 'single');
        featStruct{o} = cell([size(dataStruct.bbox,1),1]);
    end
  end

  %f = reshape(f, [feat_dim batch_size]);
  % last batch, trim f to size
  if j == length(batches)
  	for o = 1:length(cnn_model.net.outputs)
  		if batch_padding > 0
    		f{o} = f{o}(1:end-batch_padding,:,:,:,:,:,:); %hack to ensure this works for multidim outputs
    	end
  	end    
  end

  for o=1:length(cnn_model.net.outputs)
		%featStruct{o}(curr:curr+size(f{o},1)-1,:) = f{o};
		featStruct{o}(curr:curr+size(f{o},1)-1) = num2cell(f{o},[2:ndims(f{o})]);
  end
  curr = curr + batch_size;  
end
