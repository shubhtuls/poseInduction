function generatePoseFeatures(proto,suffix,inputSize,classInd,mirror,warpSquare)

globals;
dataSet = params.vpsDataset;

protoFile = fullfile(caffeProtoDir,proto,'deploy.prototxt');
binFile = fullfile(caffeModelDir,[suffix '.caffemodel']);

cnn_model=rcnn_create_model(protoFile,binFile);
cnn_model=rcnn_load_model(cnn_model);
cnn_model.cnn.input_size = inputSize;

padRatio = 0.00;
suff = '';

if(mirror==1)
    suff = 'Mirror';
end

if(warpSquare)
    cnn_model.detectors.crop_mode = 'square';
end

meanNums = [102.9801,115.9465,122.7717]; %magical numbers given by Ross
for i=1:3
    meanIm(:,:,i) = ones(inputSize)*meanNums(i);
end
cnn_model.cnn.image_mean = single(meanIm);
cnn_model.cnn.batch_size=20;

%keyboard;
saveDir = fullfile(cachedir,['rcnnPredsVps' dataSet],[proto suff]);
mkdirOptional(saveDir);
for ind = classInd
    class = pascalIndexClass(ind)
    load(fullfile(cachedir,['rotationData' dataSet],class));
    tmp.voc_image_id = {rotationData(:).voc_image_id};
    tmp.bbox = vertcat(rotationData(:).bbox);
    tmp.dataset = {rotationData(:).dataset};
    tmp.labels = ones(size(tmp.bbox,1),1);
    %keyboard;
    feat = rcnnFeaturesSingleBox(tmp,cnn_model,0,true);
    if(mirror)
        featMirror = rcnnFeaturesSingleBox(tmp,cnn_model,1,true);
	if(mirror==1)
	    feat = addFeatMirrorFeat(feat,featMirror);
	end
    end
    %keyboard;
    if(mirror==2)
	save(fullfile(saveDir,class),'feat','featMirror');
    else
	save(fullfile(saveDir,class),'feat');
    end
end

end

function feat = addFeatMirrorFeat(feat,featMirror)

permInds = [21:-1:1,22:42,63:-1:43,70:-1:64,71:77,84:-1:78];
for i=1:length(feat)
    feat{i}(1:84) = (feat{i}(1:84)+featMirror{i}(permInds))/2;
end
end
