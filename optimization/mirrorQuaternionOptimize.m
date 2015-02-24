function [r,mirrorCoeffs] = mirrorQuaternionOptimize(quats,quatsMirror)
%MIRRORQUATERNIONOPTIMIZE Summary of this function goes here
%   Detailed explanation goes here
% quats, quatsMirror are N X 4

N = size(quats,1);
mirrorCoeffs = -ones(N,1);
Mq = {};Mq_mirror = {};

for i=1:N
    Mq{i} = quatMultMatrix(quats(i,:));
    Mq_mirror{i} = diag([1 1 -1 -1])*quatMultMatrix(quatsMirror(i,:));
end

optMatrix = zeros(4*N,4);
posOptMatrix = zeros(4*N,4);
negOptMatrix = zeros(4*N,4);

for iter = 1:10
    %% updating r
    for i=1:N
        optMatrix(4*(i-1)+[1:4],:) = Mq{i} - mirrorCoeffs(i)*Mq_mirror{i};
    end
    [~,S,r] = svds(optMatrix,4);
    r = r(:,4);
    disp(S(4,4)/N)
    
    %% updating mirror coeffs
    for i=1:N
        posOptMatrix(4*(i-1)+[1:4],:) = Mq{i} - Mq_mirror{i};
    end
    errPos = reshape(posOptMatrix*r,4,N);
    errPos = sum(errPos.^2,1);
    
    for i=1:N
        negOptMatrix(4*(i-1)+[1:4],:) = Mq{i} + Mq_mirror{i};
    end
    
    errNeg = reshape(negOptMatrix*r,4,N);
    errNeg = sum(errNeg.^2,1);
    
    mirrorCoeffs = 2*(double(errPos < errNeg)-0.5);
    
end

end

function M = quatMultMatrix(q)
M = [q(1), -q(2), -q(3), -q(4);...
    q(2), q(1), -q(4), q(3);...
    q(3), q(4), q(1), -q(2);...
    q(4), -q(3), q(2), q(1)];
end