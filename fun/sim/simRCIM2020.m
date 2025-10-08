function [tH2Wn,RH2Wn,lOn] = simRCIM2020(pP,nP,lP,xmax,zmax,tO2H,RO2H,et,eR,eO,m)
%SIMRCIM2020 plane calibration simulation of https://doi.org/10.1016/j.rcim.2019.101823
%   v1.0 2025-03-26
%
%   tH2Wn - hand position in world frame with noise     (3x1xm)
%   RH2Wn - hand orientation in world frame with noise  (3x3xm)
%   lOn   - fitted intersection line in OCT frame       (6x1xm, [position direction])
%
%   pP - plane ceter position
%   nP - plane normal
%   lP - plane side length

% number of image points for each measurement
npO = 20;

% normalize plane normal
nP = nP/norm(nP);

%% noise covariance
% position
if et == 0
    Lt = zeros(3);
else
    Lt = chol(et^2/3*eye(3))';
end
% orientation
if eR == 0
    LR = zeros(3);
else
    LR = chol(eR^2/3*eye(3))';
end
% image
if eO == 0
    LO = zeros(2);
else
    LO = chol(eO^2/2*eye(2))';
end

%% orientation range
xRng = [-45 45]/180*pi;
yRng = [-45 45]/180*pi;
zRng = [0 360]/180*pi;
% eulRng = [zRng; xRng; yRng];    % Ry*Rx*Rz / 'ZXY' in eul2rotm

%% simulation
% plane orientation
if nP(1) == 0 && nP(2) == 0
    RPx = [1 0 0]';
else
    RPx = [-nP(2) nP(1) 0]';
    RPx = RPx/norm(RPx);
end
RPy = cross(nP,RPx);
RP = [RPx RPy nP];

% OCT orientation
RO2W0 = [-RPx RPy -nP];
% circular pattern as in the article
eulO2W = [linspace(zRng(1),zRng(2),m);
    linspace(xRng(1),xRng(2),m);
    linspace(yRng(1),yRng(2),m)];

% orientation noise
wn = LR*randn(3,m);
parfor i = 1:m
    Rn(:,:,i) = ExpSO3(wn(:,i));
end

% hand orientation
RH2W = pagemtimes(pagemtimes(RO2W0,eul2rotm(eulO2W','ZXY')),RO2H');
RH2Wn = pagemtimes(RH2W,Rn);

% image position & direction
pO = zeros(3,1,m);
vOn = zeros(3,1,m);
for i = 1:m
    % position
    x0 = xmax*rand;
    z0 = zmax*rand;
    pO(:,:,i) = [x0 0 z0]';

    % slope
    theta = eulO2W(3,i);
    k = tan(theta);

    % line section inside FOV
    % left border
    zl = z0 - k*x0;
    if zl < 0
        xl = x0 - z0/k;
    elseif zl > zmax
        xl = x0 + (zmax-z0)/k;
    else
        xl = 0;
    end
    % right border
    zr = z0 + k*(xmax-x0);
    if zr < 0
        xr = x0 - z0/k;
    elseif zr > zmax
        xr = x0 + (zmax-z0)/k;
    else
        xr = xmax;
    end

    % generate points for line fitting
    pOx = linspace(xl,xr,npO);
    pOz = z0 + k*(pOx-x0);

    % add noise
    % scan avergaging
    nOi = pagemtimes(LO, randn(2,npO,10));
    nOi = mean(nOi,3);
    pOxn = pOx + nOi(1,:);
    pOzn = pOz + nOi(2,:);

    % line fitting
    pn = polyfit(pOxn,pOzn,1);
    thetae = atan(pn(1));
    % fitted line direction
    vOn(:,:,i) = [cos(thetae) 0 sin(thetae)]';
end

% line position noise
% nO = pagemtimes(LO,randn(2,1,m));
% pOn = pO + [nO(1,:,:); zeros(1,1,m); nO(2,:,:)];
pOn = pO;
lOn = [pOn; vOn];

% line position in plane frame
plP = lP/2*[2*rand(2,1,m)-1; zeros(1,1,m)];

% line position in world frame
plW = pagemtimes(RP,plP) + pP;

% hand position
tH2W = plW - pagemtimes(RH2W,pagemtimes(RO2H,pO) + tO2H);
% position noise
tH2Wn = tH2W + pagemtimes(Lt,randn(3,1,m));

end

