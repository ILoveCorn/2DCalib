function [tH2Wn,RH2Wn,pcOe] = simMEAS2023(pS,rS,xMax,zMax,tO2H,RO2H,Sigmat,SigmaR,sigmaO,alphaMax,betaMax,mAlpha,mBeta,mt)
%SIMMEAS2023
%   Simulation of https://doi.org/10.1016/j.measurement.2023.112936
%
%   Input:
%       pS, rS: sphere center position, radius
%       xMax, zMax: lateral and axial FOV
%       tO2H, RO2H: hand-eye transformation
%       Sigmat, SigmaR: covariance of position and orientation noise
%       sigmaO: std of image noise
%       alphaMax, betaMax: max Euler angle
%       mAlpha, mBeta: number of alpha and beta angles
%       mt: number of measurements per orientation
%   Output:
%       tH2Wn: noisy hand-world position        (3 x mt x mR)
%       RH2Wn: noisy hand-world orientation     (3 x 3 x mR)
%       pcOe: fitted sphere center in OCT frame (3 x mt x mR)

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
RO2W0 = ExpSO3(randSO3());

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

% noisy orientation
RH2Wn = pagemtimes(RO2W,RO2H'); % 3 x 3 x mR
% actual orientation
RH2W = pagemtimes(repmat(reshape(RH2Wn,3,3,1,mR),[1 1 mt 1]),Rn);    % 3 x 3 x mt x mR

% sphere center in OCT frame
pcOx = xMax/2 + 0.25*xMax*(2*rand(1,mt)-1); % 1 x mt
pcOy = linspace(-0.8*rS,0.8*rS,mt);         % 1 x mt
rO = sqrt(rS^2 - pcOy.^2);                  % 1 x mt
% circle radius < zmax
pcOz = rO + (zMax-rO).*rand(1,mt);          % 1 x mt
% circle radius >= zmax
pcOz(rO>=zMax) = rO(rO>=zMax);
pcO = repmat([pcOx; pcOy; pcOz],[1 1 mR]);  % 3 x mt x mR

% sphere points
npO = 30;
pO = zeros(3,npO,mt,mR);
for it = 1:mt
    % left border
    xl = max(pcOx(it)-rO(it),0) + 1e-4;
    % right border
    xr = min(pcOx(it)+rO(it),zMax) - 1e-4;

    % image points
    pOx = linspace(xl,xr,npO);                          % 1 x npO
    pOz = pcOz(it) - sqrt(rO(it)^2-(pOx-pcOx(it)).^2);  % 1 x npO
    pO(:,:,it,:) = [repmat(pOx,[1 1 1 mR]);
        zeros(1,npO,1,mR);
        repmat(pOz,[1 1 1 mR])];
end

% image noise
nO = pagemtimes(LO,randn(2,npO,mt,mR));
pOn = pO + [nO(1,:,:,:); zeros(1,npO,mt,mR); nO(2,:,:,:);];   % 3 x npO x mt x m

% sphere fitting
pcOe = zeros(3,mt,mR);
for iR = 1:mR
    for it = 1:mt
        pOnji = pOn(:,:,it,iR);   % 3 x npO
        [pcOex,pcOez,rOe] = circleFit(pOnji(1,:),pOnji(3,:));
        % handle fitting error
        if rOe > rS
            rOe = rS;
        end
        pcOey = sign(pcOy(it))*sqrt(rS^2-rOe^2);
        pcOe(:,it,iR) = [pcOex pcOey pcOez]';
    end
end

% pre-computing
pcH = pagemtimes(RO2H,reshape(pcO,3,1,mt,mR)) + tO2H;  % 3 x 1 x mt x m

% hand position
tH2W = repmat(pS,[1 1 mt mR]) - pagemtimes(RH2W,pcH);
% position noise
tH2Wn = reshape(tH2W,3,mt,mR) + pagemtimes(Lt,randn(3,mt,mR));    % 3 x mt x m

end

