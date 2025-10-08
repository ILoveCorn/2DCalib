function [nP,dP,res] = planeFit(tO2H,RO2H,tH2W,RH2W,scanO)
%PLANEFIT Estimate plane parameters (nP,dP) from hand-eye
%transformation and plane calibration data
%   tO2H, RO2H - hand-eye transformation
%   tH2W, RH2W, scanO - plane calibration data
%
%   nP - plane normal
%   dP - plane distance
%   res - averaged plane residue

% transform to world frame
scanW = pagemtimes(RH2W, pagemtimes(RO2H, scanO) + tO2H) + tH2W; % 3 x np x m
scanW = reshape(scanW, 3, []);  % 3 x np*m

% estimate normal using SVD
scanWavg = mean(scanW, 2);  % 3 x 1
[~,~,V] = svd((scanW-scanWavg)', 0);
nP = V(:, 3);

% plane distance
dP = nP'*scanWavg;

% averaged L1 norm
res = norm(nP'*scanW-dP,1) / size(scanW, 2);
end

