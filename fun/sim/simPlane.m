function [tH2Wn,RH2Wn,pOn,vOn,scanO] = simPlane(pP,nP,lP,xMax,zMax,tO2H,RO2H,Sigmat,SigmaR,sigmaO,alphaMax,betaMax,mAlpha,mBeta)
%SIMPLANE
%   Plane calibration simulation
%
%   Input:
%       pP, nP, lP: plane center position, normal, side length
%       xMax, zMax: lateral and axial FOV
%       tO2H, RO2H: hand-eye transformation
%       Sigmat, SigmaR: covariance of position and orientation noise
%       sigmaO: std of image noise
%       alphaMax, betaMax: max Euler angle
%       mAlpha, mBeta: number of alpha and beta angles
%   Output:
%       tH2Wn: noisy hand-world position            (3 x 1 x m)
%       RH2Wn: noisy hand-world orientation         (3 x 3 x m)
%       pOn: intersection position in OCT frame     (3 x 1 x m)
%       vOn: intersection direction in OCT frame    (3 x 1 x m)
%       scanO: intersection points in OCT frame     (3 x npO x m)

% number of measurements
m = mAlpha*mBeta;

% number of image points for each measurement
npO = 20;

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

% plane orientation
if nP(1) == 0 && nP(2) == 0
    RPx = [1 0 0]';
else
    RPx = [-nP(2) nP(1) 0]';
    RPx = RPx/norm(RPx);
end
RPy = cross(nP,RPx);
RP = [RPx RPy nP];

% initial OCT orientation
RO2W0 = [-RPx RPy -nP];

% sampling Euler angles
[eulx, euly] = ndgrid(linspace(-betaMax,betaMax,mBeta), linspace(-alphaMax,alphaMax,mAlpha));
eulx = eulx(:);
euly = euly(:);
eulO2W = [zeros(size(eulx)) euly eulx];

% OCT orientation
RO2W = pagemtimes(RO2W0,eul2rotm(eulO2W,'ZYX'));

% orientation noise
Rn = zeros(3,3,m);
wn = LR*randn(3,m);
for i = 1:m
    Rn(:,:,i) = ExpSO3(wn(:,i));
end

% hand orientation
RH2W = pagemtimes(RO2W,RO2H');
RH2Wn = pagemtimes(RH2W,Rn);

% intersection in OCT
pO = zeros(3,1,m);
pOn = zeros(3,1,m);
vOn = zeros(3,1,m);
scanO = zeros(3,npO,m);
for i = 1:m
    % line position
    x0 = xMax*rand;
    z0 = zMax*rand;
    pO(:,:,i) = [x0 0 z0]';

    % y-axis of OCT frame
    RO2Wy = RO2W(:,2,i);

    % intersection direction
    A = RO2Wy'*RPx;
    B = RO2Wy'*RPy;

    if abs(A) < 1e-10
        vW = RPx;
    elseif abs(B) < 1e-10
        vW = RPy;
    else
        vW = RPx - A/B*RPy;
    end
    vO = RO2W(:,:,i)'*vW;

    % slope
    k = vO(3)/vO(1);

    % line segment inside FOV
    % left border
    zl = z0 - k*x0;
    if zl < 0
        xl = x0 - z0/k;
    elseif zl > zMax
        xl = x0 + (zMax-z0)/k;
    else
        xl = 0;
    end
    % right border
    zr = z0 + k*(xMax-x0);
    if zr < 0
        xr = x0 - z0/k;
    elseif zr > zMax
        xr = x0 + (zMax-z0)/k;
    else
        xr = xMax;
    end

    % generate points for line fitting
    pOx = linspace(xl,xr,npO);
    pOz = z0 + k*(pOx-x0);

    % averaged scan noise
    nOi = pagemtimes(LO,randn(2,npO,10));
    nOi = mean(nOi,3);
    % add noise
    pOxn = pOx + nOi(1,:);
    pOzn = pOz + nOi(2,:);
    scanO(1,:,i) = pOxn;
    scanO(3,:,i) = pOzn;

    % line position with noise
    pOn(:,:,i) = [mean(pOxn) 0 mean(pOzn)]';

    % line fitting
    pn = polyfit(pOxn,pOzn,1);
    thetae = atan(pn(1));
    % fitted line direction
    vOn(:,:,i) = [cos(thetae) 0 sin(thetae)]';
end

% line position in plane frame
plP = lP/2*[2*rand(2,1,m)-1; zeros(1,1,m)];

% line position in world frame
pW = pagemtimes(RP,plP) + pP;

% hand position
tH2W = pW - pagemtimes(RH2W,pagemtimes(RO2H,pO) + tO2H);
% add position noise
tH2Wn = tH2W + pagemtimes(Lt,randn(3,1,m));

% DEBUG
% pW = reshape(pagemtimes(RH2W, pagemtimes(RO2H, pOn) + tO2H) + tH2W, 3, []);
% vW = reshape(pagemtimes(RH2W, pagemtimes(RO2H, vOn)), 3, []);
% % X = pW(1, :);
% % Y = pW(2, :);
% % Z = pW(3, :);
% X = zeros(size(pW(1, :)));
% Y = zeros(size(pW(2, :)));
% Z = zeros(size(pW(3, :)));
% U = vW(1, :);
% V = vW(2, :);
% W = vW(3, :);
% quiver3(X, Y, Z, U, V, W);
% grid on
% axis equal

end

