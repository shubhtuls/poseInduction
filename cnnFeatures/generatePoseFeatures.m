function generatePoseFeatures(proto,suffix,inputSize,classInd,mirror,warpSquare, similarClassInd)

globals;
dataSet = params.vpsDataset;

protoFile = fullfile(caffeProtoDir,proto,'deploy.prototxt');
binFile = fullfile(caffeModelDir,suffix, 'netFinal.caffemodel');

cnn_model=cnn_create_model(protoFile,binFile);
cnn_model=cnn_load_model(cnn_model);
cnn_model.input_size = inputSize;

suff = '';

if(~exist('similarClassInd','var'))
    similarClassInd = classInd;
end
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
cnn_model.image_mean = single(meanIm);
cnn_model.batch_size=20;

%keyboard;
saveDir = fullfile(cachedir,['rcnnPredsVps' dataSet],[proto suff]);
mkdirOptional(saveDir);
for i = 1:length(classInd)
    ind = classInd(i);
    simInd = similarClassInd(i);
    class = pascalIndexClass(ind,dataSet)
    load(fullfile(cachedir,['rotationData' dataSet],class));
    tmp.voc_image_id = {rotationData(:).voc_image_id};
    tmp.bbox = vertcat(rotationData(:).bbox);
    tmp.dataset = {rotationData(:).dataset};
    tmp.labels = ones(size(tmp.bbox,1),1)*simInd;
    %keyboard;
    
    featStruct = cnnFeaturesSingleBox(tmp,cnn_model,0,true);
    outNames = cnn_model.net.outputs;
    if(mirror)
        featStructMirror = cnnFeaturesSingleBox(tmp,cnn_model,1,true);
    end
    if(mirror==1)
        featStruct = addFeatMirrorFeat(featStruct,featStructMirror,outNames);
    end

    %keyboard;
    if(mirror==2)
    	save(fullfile(saveDir,class),'featStruct','outNames','featStructMirror');
    else
        save(fullfile(saveDir,class),'featStruct','outNames');
    end
end

end

function feat = addFeatMirrorFeat(feat,featMirror,outNames)

for o=1:length(outNames)
	if(strcmp(outNames{o},'poseClassify'))
		permInds = [21:-1:1,22:42,63:-1:43,70:-1:64,71:77,84:-1:78];
        for i = 1:length(feat{o})
    		feat{o}{i}(1:84) = (feat{o}{i}(1:84)+featMirror{o}{i}(permInds))/2;
        end
	end
end

end