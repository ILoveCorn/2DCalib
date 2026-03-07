function [tO2He,RO2He,time] = sphere_MEAS2023(tH2Wn,RH2Wn,pcOe)
%SPHERE_MEAS2023
%   Implementation of https://doi.org/10.1016/j.measurement.2023.112936
%
%   Input:
%       tH2Wn: hand-world position with noise       (3 x mt x m)
%       RH2Wn: hand-world orientation with noise    (3 x 3 x m)
%       pcOe: fitted sphere center in OCT frame     (3 x mt x m)
%   Output:
%       tO2He: estimated hand-eye position          (3 x 1)
%       RO2He: estimated hand-eye orientation       (3 x 3)
%       time: computation time

% number of measurements
mt = size(tH2Wn,2);
m = size(tH2Wn,3);

tic % start clock
%%% orientation %%%
% build equation
Ap = pcOe(:,1,:) - pcOe(:,2:end,:);
A = reshape(Ap,3,m*(mt-1));
Bp = pagemtimes(RH2Wn,'transpose',tH2Wn(:,2:end,:)-tH2Wn(:,1,:),'none');
B = reshape(Bp,3,m*(mt-1));

% SVD
[U,~,V] = svd(A*B');
RO2He = V*U';

% check determinant of R
if abs(det(RO2He)-1) > 1e-5
    V(:,3) = -V(:,3);
    RO2He = V*U';
end

%%% position %%%
Cpi = RH2Wn(:,:,1) - RH2Wn(:,:,2:end);                  % 3*3*(m-1)
Cp = repmat(reshape(Cpi,3,3,1,m-1),[1 1 mt 1]);         % 3*3*mt*(m-1)
C = reshape(permute(Cp,[1 3 4 2]),3*mt*(m-1),3);        % 3mt(m-1)*3
Dpi = pagemtimes(RH2Wn,pagemtimes(RO2He,pcOe)) + tH2Wn; % 3*mt*m
Dp = Dpi(:,:,2:end) - Dpi(:,:,1);                       % 3*mt*(m-1)
D = reshape(Dp,3*mt*(m-1),1);                           % 3mt(m-1)*1
tO2He = C\D;

% time used
time = toc;

end
