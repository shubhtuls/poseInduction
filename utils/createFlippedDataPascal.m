function [flipData] = createFlippedDataPascal(data,kpsPerm)
%CREATEFLIPPEDDATA Summary of this function goes here
%   Detailed explanation goes here

% kps are stored as x,y
N = length(data);
globals;

flipData = [data data];

for i=1:N
    
    imX = data(i).imsize(2);
    bbox = data(i).bbox;
    
    %% permute and flip kps
    if(~isempty(flipData(i).kps))
        kps = flipData(i).kps(kpsPerm,:);
        kps(:,1) = imX-kps(:,1) + 1;
        flipData(N+i).kps = kps;
    else
        kps = nan(size(data(1).kps));
        flipData(N+i).kps = kps;
        flipData(i).kps = kps;
    end
    %% flip bbox [x1 y1 x2 y2]
    flipData(N+i).bbox([1 3]) = imX + 1 - bbox([3 1]);
    
end

end

