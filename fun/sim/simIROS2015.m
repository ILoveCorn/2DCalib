function [tH2Wn,RH2Wn,pOn] = simIROS2015(nPlane,pPs,nPs,lPs,xMax,zMax,tO2H,RO2H,Sigmat,SigmaR,sigmaO,alphaMax,betaMax,mAlpha,mBeta)
%SIMIROS2015
%   Multi-plane calibration simulation of https://doi.org/10.1109/IROS.2015.7353884
%
%   Input:
%       nPlane: number of planes
%       pPs, nPs, lPs: plane center position, normal, side length
%       xMax, zMax: lateral and axial FOV
%       tO2H, RO2H: hand-eye transformation
%       Sigmat, SigmaR: covariance of position and orientation noise
%       sigmaO: std of image noise
%       alphaMax, betaMax: max Euler angle
%       mAlpha, mBeta: number of alpha and beta angles
%   Output:
%       tH2Wn: noisy hand-world position            (3 x 1 x mi x nPlane)
%       RH2Wn: noisy hand-world orientation         (3 x 3 x mi x nPlane)
%       pOn: intersection position in OCT frame     (3 x 1 x mi x nPlane)

mAlphai = round(mAlpha/sqrt(nPlane));
mBetai = round(mBeta/sqrt(nPlane));
mi = mAlphai*mBetai;
tH2Wn = zeros(3,1,mi,nPlane);
RH2Wn = zeros(3,3,mi,nPlane);
pOn = zeros(3,1,mi,nPlane);

for i = 1:nPlane
    [tH2Wn(:,:,:,i),RH2Wn(:,:,:,i),pOn(:,:,:,i),~,~] = simPlane(pPs(:,i),nPs(:,i),lPs(:,i),xMax,zMax,tO2H,RO2H,Sigmat,SigmaR,sigmaO,alphaMax,betaMax,mAlphai,mBetai);
end

end

