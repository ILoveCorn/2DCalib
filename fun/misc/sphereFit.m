function [pCf,Df] = sphereFit(tO2H,RO2H,tH2W,RH2W,scanO)
%SPHEREFIT
%   Sphere fitting wrapper of pcfitsphere
%
%   Input:
%       tO2H: hand-eye position         (3 x 1)
%       RO2H: hand-eye orientation      (3 x 3)
%       tH2W: hand-world position       (3 x 1 x m)
%       RH2W: hand-world orientation    (3 x 3 x m)
%       scanO: target profile points    ({3 x mpi} x m)
%   Output:
%       pCf: fitted sphere center       (3 x 1)
%       Df: fitted sphere diameter      (1 x 1)

% number of measurements
m = size(tH2W,3);

% transform profile points
scanW = cell(1,m);
for iScan = 1:m
    scanW{iScan} = RH2W(:,:,iScan)*(RO2H*scanO{iScan} + tO2H) + tH2W(:,:,iScan);
end
scanW = [scanW{:}]';

% sphere fitting
ptCloud = pointCloud(scanW);
warning('off')
[sphereModel, ~, ~, ~] = pcfitsphere(ptCloud, 0.1);
warning('on')

pCf = sphereModel.Center';
Df = 2*sphereModel.Radius;
end

