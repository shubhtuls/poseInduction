function [pos] = readDataIlsvrc(classInd)
%READDATAILSVRC Summary of this function goes here
%   Detailed explanation goes here

globals;

%% File names

pos = [];
load(fullfile(ilsvrcDir,'imdbs',['imdb_ilsvrc13_train_pos_' num2str(classInd)]))
load(fullfile(ilsvrcDir,'roidbs',['roidb_ilsvrc13_train_pos_' num2str(classInd)]))
cls = imdb.classes{1};
classPrefix = imdb.image_ids{1}(1:9);
%% Create one entry per bounding box in the pos array
numpos = 0;
vp.azimuth=pi; vp.elevation=0; vp.theta=0;

for j = find(~imdb.is_blacklisted')
    classPrefix = imdb.image_ids{j}(1:9);
    for k=find((roidb.rois(j).class')==classInd)
        numpos = numpos + 1;
        bbox   = roidb.rois(j).boxes(k,:);
        imgsize = imdb.sizes(j,:);
        if(imgsize(1) < min(bbox([2 4])) || imgsize(2) < min(bbox([1 3])))
            bbox = [1 1 imgsize([2 1])];
        end

        pos(numpos).imsize = imgsize;
        pos(numpos).voc_image_id = fullfile(classPrefix,imdb.image_ids{j});
        pos(numpos).voc_rec_id = k;

        pos(numpos).bbox   = bbox;
        pos(numpos).view    = '';
        pos(numpos).kps     = [];
        pos(numpos).part_names  = {};

        pos(numpos).poly_x      = [];
        pos(numpos).poly_y      = [];
        pos(numpos).mask = [];
        pos(numpos).class       = cls;
        pos(numpos).rot = [];
        pos(numpos).euler = [];
        pos(numpos).detScore = Inf;
        pos(numpos).IoU = 1;

        %% Getting camera
        objectInd = 1;%k
        [rot,euler]=viewpointToRots(vp);
        pos(numpos).rot=rot;
        pos(numpos).euler=euler;
        pos(numpos).objectInd = objectInd;
        pos(numpos).dataset = 'ilsvrc';
        pos(numpos).subtype = 1;

    end
end

end


function [R,euler] = viewpointToRots(vp)
    euler = [vp.azimuth vp.elevation vp.theta]' .* pi/180;
    R = angle2dcm(euler(3), euler(2)-pi/2, -euler(1),'ZXZ'); %took a lot of work to figure this formula out !!
    euler = euler([3 2 1]);
end