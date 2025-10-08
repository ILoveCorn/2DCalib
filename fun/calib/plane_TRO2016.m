function [tO2He,RO2He,ne,time] = plane_TRO2016(tO2He,RO2He,tH2Wn,RH2Wn,vOn,scanO)
%PLANE_TRO2016 implementation of https://doi.org/10.1109/TRO.2016.2596769
%   Scheme: decoupled rotation & translation, PCA
%   Parameterization: <RPY,R(3)>
%   Solver: nonlinear least square
%   v1.0 2025-06-08
%
%   tH2Wn - hand position in world frame with noise     (3 x 1 x m)
%   RH2Wn - hand orientation in world frame with noise  (3 x 3 x m)
%   vOn   - fitted line direction in OCT frame          (3 x 1 x m)
%   scanO - original line points                        (3 x np x m)

% start clock
tic

%%% rotation %%%
% initial hand-eye rotation
eulO2He = rotm2eul(RO2He);
% solve
options = optimoptions('lsqnonlin','Display','none');
eulO2He = lsqnonlin(@(x)costRotation(x,RH2Wn,vOn),eulO2He,[],[],options);
RO2He = eul2rotm(eulO2He);

% estimate plane normal
ne = getPlaneNormal(RH2Wn,RO2He,vOn);

%%% translation %%%
options = optimoptions('lsqnonlin','Display','none','PlotFcn',[]);
tO2He = lsqnonlin(@(x)costTranslation(x,RO2He,ne,tH2Wn,RH2Wn,scanO),tO2He,[],[],options);

time = toc;

end

%% estimate plane normal
function ne = getPlaneNormal(RH2Wn,RO2He,vOn)
% reprojection
vWe = pagemtimes(RH2Wn,pagemtimes(RO2He,vOn));  % 3 x 1 x m

% estimate normal using SVD
[~,~,V] = svd(reshape(vWe,3,[])',0);
ne = V(:,3);
end

%% rotation cost
function F = costRotation(x,RH2Wn,vOn)
% hand-eye rotation matrix
RO2He = eul2rotm(x);

% reprojection
vWe = pagemtimes(RH2Wn,pagemtimes(RO2He,vOn));

% estimate plane normal
ne = getPlaneNormal(RH2Wn,RO2He,vOn);

% return cost
F = reshape(vWe,3,[])'*ne;  % m x 1
end

%% translation cost
function F = costTranslation(x,RO2He,ne,tH2Wn,RH2Wn,scanO)
% reprojection
scanWe = pagemtimes(RH2Wn, pagemtimes(RO2He,scanO) + x) + tH2Wn;    % 3 x np x m

% return cost
F = reshape(scanWe,3,[])'*ne;   % m x 1
F = F - mean(F);
end