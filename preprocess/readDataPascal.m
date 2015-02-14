function pos = readDataPascal(cls,year,occludedTruncated)
% Read pascal to generate filtered instances of classes
% Also reads PASCAL3D dataset to figure out the object index in pascal3D

globals;

dataset_fg = 'trainval';
conf.year=year;

conf.dev_kit = pascalDir;
VOCopts    = get_voc_opts(conf);
load(fullfile(annotationDir,'segkps',cls));

if(nargin < 3)
    occludedTruncated = 0;
end

% Setup keypoints and segmentations voc_id
kps_voc_id = cellfun(@(x,y) [x, '_', y], keypoints.voc_image_id, arrayfun(@num2str, keypoints.voc_rec_id, 'Uniform', false), 'UniformOutput', false);
segs_voc_id = cellfun(@(x,y) [x, '_', y], segmentations.voc_image_id, arrayfun(@num2str, segmentations.voc_rec_id, 'Uniform', false), 'UniformOutput', false);

  % Positive examples from the foreground dataset
  ids      = textread(sprintf(VOCopts.imgsetpath, dataset_fg), '%s');
  pos      = [];
  numpos   = 0;
  for i = 1:length(ids);
    if(mod(i,500)==0)
        fprintf('Parsed ids %d/%d\n',i,length(ids));
    end
    % Parse record and exclude difficult examples
    rec           = PASreadrecord(sprintf(VOCopts.annopath, ids{i}));
    clsinds       = strmatch(cls, {rec.objects(:).class}, 'exact');
    count         = length(clsinds(:));
    % Skip if there are no objects in this image
    if count == 0
      continue;
    end

    diff          = [rec.objects(clsinds).difficult];
    trunc         = [rec.objects(clsinds).truncated];
    if(~isempty(clsinds) && isfield(rec.objects(clsinds(1)),'occluded'))
        occl = [rec.objects(clsinds).occluded];
    else
        occl = true(size(clsinds));
    end
    if(occludedTruncated==1)
        clsinds2del = diff | ~(trunc | occl); % Change this for different datasets
    elseif(occludedTruncated==-1)
        clsinds2del = diff;
    else
        clsinds2del = diff | trunc | occl;
    end
    %clsinds2del = diff;

    clsinds(clsinds2del)=[];

    count         = length(clsinds(:));
    % Skip if there are no objects in this image
    if count == 0
      continue;
    end

    % Create one entry per bounding box in the pos array
    for j = clsinds(:)'
      numpos = numpos + 1;
      j_voc_id = [rec.filename(1:end-4) '_' num2str(j)];
      ki = find(ismember(kps_voc_id,j_voc_id));
      si = find(ismember(segs_voc_id,j_voc_id));
      bbox   = rec.objects(j).bbox;
      pos(numpos).imsize = [rec.size.height rec.size.width];
      pos(numpos).voc_image_id = rec.filename(1:end-4);
      pos(numpos).voc_rec_id = j;
      
      pos(numpos).bbox   = bbox;
      pos(numpos).view    = rec.objects(j).view;
      pos(numpos).kps     = squeeze(keypoints.coords(ki,:,:));
      pos(numpos).part_names  = keypoints.labels;

      pos(numpos).poly_x      = segmentations.poly_x{si};
      pos(numpos).poly_y      = segmentations.poly_y{si};
      pos(numpos).mask = [];
      if(~isempty(pos(numpos).poly_x))
          pos(numpos).mask = roipoly(zeros(pos(numpos).imsize),pos(numpos).poly_x,pos(numpos).poly_y);
      end
      pos(numpos).class       = cls;
      pos(numpos).rot = [];
      pos(numpos).euler = [];
      pos(numpos).detScore = Inf;
      pos(numpos).subtype = 0;
      pos(numpos).IoU = 1;

      %% Getting camera
      pascal3Dfile = fullfile(PASCAL3Ddir,'Annotations',[cls '_pascal'],[pos(numpos).voc_image_id '.mat']); 
      if(exist(pascal3Dfile,'file'))
          record = load(pascal3Dfile);record = record.record;
          objectInd = 0;
          for kk=1:length(record.objects)
                if(~isempty(record.objects))
                      if(sum(bbox == record.objects(kk).bbox)==4)
                          objectInd = kk;
                      end
                end
          end
          if(objectInd)
              viewpoint = record.objects(objectInd).viewpoint;
              [rot,euler]=viewpointToRots(viewpoint);
              pos(numpos).rot=rot;
              pos(numpos).euler=euler;
              pos(numpos).subtype = record.objects(objectInd).cad_index;
          end
          pos(numpos).objectInd = objectInd;
          pos(numpos).dataset = 'pascal';
          %disp('blah')
      end
    end
  end

end

function VOCopts = get_voc_opts(conf)
% cache VOCopts from VOCinit
persistent voc_opts;

key = conf.year;
if isempty(voc_opts) || ~voc_opts.isKey(key)
  if isempty(voc_opts)
    voc_opts = containers.Map();
  end
  tmp = pwd;
  cd(conf.dev_kit);
  addpath([cd '/VOCcode']);
  VOCinit;
  cd(tmp);
  voc_opts(key) = VOCopts;
end
VOCopts = voc_opts(key);
end


function [R,euler] = viewpointToRots(vp)
        euler = [vp.azimuth vp.elevation vp.theta]' .* pi/180;
        R = angle2dcm(euler(3), euler(2)-pi/2, -euler(1),'ZXZ'); %took a lot of work to figure this formula out !!
        euler = euler([3 2 1]);
end
