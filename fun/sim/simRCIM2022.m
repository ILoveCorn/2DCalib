function [tH2Wn,RH2Wn,pOn] = simRCIM2022(pL,vL,lL,xMax,zMax,tO2H,RO2H,Sigmat,SigmaR,sigmaO,alphaMax,betaMax,mAlpha,mBeta,mt)
%SIMRCIM2022 simulation for https://doi.org/10.1016/j.rcim.2021.102221
%
%   Input:
%       pL, vL, lL: line starting point, direction, length
%       xMax, zMax: lateral, axial FOV
%       tO2H, RO2H: hand-eye transformation
%       sigmat, sigmaR, sigmaO: std of position, orientation, image noise
%       Sigmat, SigmaR: covariance of position, orientation noise
%       sigmaO: std of image noise
%       alphaMax, betaMax: max Euler angle
%       mAlpha, mBeta: number of alpha, beta angles
%       mt: number of measurements per orientation
%   Output:
%       tH2Wn - noisy hand-world position           (3 x mt x mR)
%       RH2Wn - noisy hand-world orientation        (3 x 3 x mR)
%       pOn   - intersection position in OCT frame  (3 x mt x mR)

% Cholesky decomposition for generating Gaussian noise
% % position
% if sigmat == 0
%     Lt = zeros(3);
% else
%     Lt = chol(sigmat^2*eye(3))';
% end
% % orientation
% if sigmaR == 0
%     LR = zeros(3);
% else
%     LR = chol(sigmaR^2*eye(3))';
% end

% position
Lt = chol(Sigmat)';
% orientation
LR = chol(SigmaR)';
% image
if sigmaO == 0
    LO = zeros(2);
else
    LO = chol(sigmaO^2*eye(2))';
end

% number of orientations
mR = mAlpha*mBeta;

% initial OCT orientation
if vL(1) == 0 && vL(2) == 0
    RO2W0x = [1 0 0]';
else
    RO2W0x = [-vL(2) vL(1) 0]';
    RO2W0x = RO2W0x/norm(RO2W0x);
end
RO2W0z = cross(RO2W0x,vL);
RO2W0 = [RO2W0x vL RO2W0z];

% sampling Euler angles
[eulx, euly] = ndgrid(linspace(-betaMax,betaMax,mBeta), linspace(-alphaMax,alphaMax,mAlpha));
eulx = eulx(:);
euly = euly(:);
eulO2W = [zeros(size(eulx)) euly eulx];

% OCT orientation
RO2W = pagemtimes(RO2W0,eul2rotm(eulO2W,'ZYX'));

% orientation noise
Rn = zeros(3,3,mt,mR);
wn = pagemtimes(LR,randn(3,mt,mR));
for iR = 1:mR
    for it = 1:mt
        Rn(:,:,it,iR) = ExpSO3(wn(:,it,iR));
    end
end

% % OCT orientation
% RO2W = zeros(3,3,mR);
% parfor iR = 1:mR
%     RO2Wi = eye(3);
%     Ry = zeros(3,1);
%     % check if line target is parallel to image plane
%     while vL'*Ry == 0
%         RO2Wi = ExpSO3(randSO3());
%         Ry = RO2Wi(:,2);
%     end
%     RO2W(:,:,iR) = RO2Wi;
%
%     % orientation noise
%     for it = 1:mt
%         Rn(:,:,it,iR) = ExpSO3(wn(:,it,iR));
%     end
% end

% noisy orientation
RH2Wn = pagemtimes(RO2W,RO2H'); % 3 x 3 x mR
% actual orientation
RH2W = pagemtimes(repmat(reshape(RH2Wn,3,3,1,mR),[1 1 mt 1]),Rn); % 3 x 3 x mt x mR

% intersection in world frame
pW = pL + pagemtimes(vL*lL,rand(1,1,mt,mR)); % 3 x 1 x mt x mR

% intersection in OCT frame
pO = [xMax; 0; zMax].*[rand(1,mt,mR); zeros(1,mt,mR); rand(1,mt,mR)]; % 3 x mt x mR

% add image noise
nO = pagemtimes(LO,randn(2,mt,mR));
pOn = pO + [nO(1,:,:); zeros(1,mt,mR); nO(2,:,:)]; % 3 x mt x mR

% hand position
tH2W = pW - pagemtimes(RH2W,pagemtimes(RO2H,reshape(pO,3,1,mt,mR)) + tO2H); % 3 x 1 x mt x mR
% add position noise
tH2Wn = reshape(tH2W,3,mt,mR) + pagemtimes(Lt,randn(3,mt,mR));

end

