% sim_accuracy_vs_VFE.m
%   test calibration accuracy with and without variance factor estimation

clearvars
close all
clc

rng('default')

%% Simulation settings
% sensor field-of-view
xMax = 5;
zMax = 2;

% variance factors
sigmat = 1;                 % position (mm)
sigmaRd = 0.1;              % orientation (degree)
sigmaR = deg2rad(sigmaRd);  % orientation (rad)
sigmaO = 0;                 % image (mm)

% noise covariance isotropic
Sigmat = sigmat^2*eye(3);
SigmaR = sigmaR^2*eye(3);

% OCT orientation range
alphaMax = deg2rad(45);
betaMax = deg2rad(45);

% number of Euler angles
mAlpha = 10;
mBeta = mAlpha;
% number of measurements
m = mAlpha*mBeta;

% repeated times
nTest = 100;

% intial value deviation
dt = 5;                 % hand-eye position (mm)
dTheta = deg2rad(5);    % hand-eye orientation (rad)

%% Method settings
% max iterations for proposed method
iImax = 30;

% convergence tolerance
tol = 1e-5;

%% Simulation
ePos = zeros(nTest,3);
eRot = zeros(nTest,3);
allSigmae = zeros(nTest,2);

for iTest = 1:nTest
    fprintf('Repeat test: %d / %d\n', iTest, nTest)

    % random hand-eye transformation
    tO2H = 50*(2*rand(3,1)-1);
    RO2H = ExpSO3(randSO3());

    % plane parameters
    pP = zeros(3,1);    % center point
    nP = rand(3,1);
    nP = nP/norm(nP);   % direction
    lP = 50;            % side length

    % initial value
    tO2H0 = tO2H + dt*(2*rand(3,1)-1);
    RO2H0 = RO2H*axang2rotm([rand(1,3) dTheta*(2*rand-1)]);
    nP0 = nP + 0.05*(2*rand(3,1)-1);
    nP0 = nP0 / norm(nP0);
    dP0 = pP'*nP0;

    % simulated data
    [tH2Wn,RH2Wn,pOn,vOn,scanO] = simPlane(pP,nP,lP,xMax,zMax,tO2H,RO2H,Sigmat,SigmaR,sigmaO,alphaMax,betaMax,mAlpha,mBeta);

    % 1 - w/ VFE
    iOmax = 50;

    % random initial variance factor
    kMin = 0.75;
    kMax = 1.25;
    kt = kMin + (kMax-kMin)*rand;
    kR = kMin + (kMax-kMin)*rand;
    sigma0 = [kt*sigmat kR*sigmaR];

    [tO2He,RO2He,~,~,~,~,sigmae,~] = plane_Jnt_GH(tO2H0,RO2H0,nP0,dP0,sigma0,tH2Wn,RH2Wn,pOn,vOn,iImax,iOmax,tol);
    [ePos(iTest,1),eRot(iTest,1)] = calibrationError(tO2H,RO2H,tO2He,RO2He);
    allSigmae(iTest,:) = sigmae;

    % 2 - w/o VFE (ground truth)
    iOmax = 1;
    sigma0 = [sigmat sigmaR];
    [tO2He,RO2He,~,~,~,~,~,~] = plane_Jnt_GH(tO2H0,RO2H0,nP0,dP0,sigma0,tH2Wn,RH2Wn,pOn,vOn,iImax,iOmax,tol);
    [ePos(iTest,2),eRot(iTest,2)] = calibrationError(tO2H,RO2H,tO2He,RO2He);

    % 3 - w/o VFE (equal weights)
    iOmax = 1;
    sigma0 = ones(1,2);
    [tO2He,RO2He,~,~,~,~,~,~] = plane_Jnt_GH(tO2H0,RO2H0,nP0,dP0,sigma0,tH2Wn,RH2Wn,pOn,vOn,iImax,iOmax,tol);
    [ePos(iTest,3),eRot(iTest,3)] = calibrationError(tO2H,RO2H,tO2He,RO2He);
end

%% Boxplot
labels = {'VFE'; 'GT'; 'EQ'};

fig = figure(1);
set(fig, 'Units','centimeters', 'Position',[2 2 8.8 6.6], 'Color','w');

tiledlayout(1,2,"TileSpacing","compact","Padding","compact")

% eRot
nexttile
bRot = boxplot(eRot,'Labels',labels);
grid on
box on
ylabel('$\varepsilon_{\mathbf{R}} \; [^\circ]$','Interpreter','latex')
set(gca,'FontName','Times New Roman','FontSize',9,'TickLabelInterpreter','latex');

set(findobj(bRot,'Tag','Median'),'LineWidth',2.0);
set(findobj(bRot,'Tag','Box'),'LineWidth',1.2);
set(findobj(bRot,'Tag','Whisker'),'LineWidth',1.2);

% ePos
nexttile
bPos = boxplot(ePos,'Labels',labels);
grid on
box on
ylabel('$\varepsilon_{\mathbf{t}} \; [\mathrm{mm}]$','Interpreter','latex')
set(gca,'FontName','Times New Roman','FontSize',9,'TickLabelInterpreter','latex');

set(findobj(bPos,'Tag','Median'),'LineWidth',2.0);
set(findobj(bPos,'Tag','Box'),'LineWidth',1.2);
set(findobj(bPos,'Tag','Whisker'),'LineWidth',1.2);

%% Display accuracy of estimated variance factors
% Ground truth (convert σ_R to degrees for reporting)
sigma_gt = [sigmat, sigmaR];
sigma_gt_deg = [sigmat, sigmaRd];  % for display in degrees
allSigmae_deg = [allSigmae(:,1), rad2deg(allSigmae(:,2))];  % convert estimated σ_R to degrees

% Estimation error (in degrees for orientation)
sigmaError_deg = allSigmae_deg - sigma_gt_deg;
sigmaErrorRel = 100 * sigmaError_deg ./ sigma_gt_deg;  % relative error (%)

% Statistics
fprintf('\n');
fprintf('================================================================================\n');
fprintf('              Accuracy of Variance Factor Estimation (VFE)\n');
fprintf('================================================================================\n');
fprintf('Ground truth:  σ_t = %.4f mm,  σ_R = %.4f°\n', sigmat, sigmaRd);
fprintf('--------------------------------------------------------------------------------\n');
fprintf('                              σ_t (mm)           σ_R (°)\n');
fprintf('--------------------------------------------------------------------------------\n');
fprintf('Ground truth:              %10.4f          %10.4f\n', sigma_gt_deg(1), sigma_gt_deg(2));
fprintf('Estimated (median(IQR)): %7.4f (%.4f)    %7.4f (%.4f)\n', ...
    median(allSigmae_deg(:,1)), iqr(allSigmae_deg(:,1)), median(allSigmae_deg(:,2)), iqr(allSigmae_deg(:,2)));
fprintf('--------------------------------------------------------------------------------\n');
fprintf('Absolute error (median(IQR)): %7.4f (%.4f)    %7.4f (%.4f)\n', ...
    median(sigmaError_deg(:,1)), iqr(sigmaError_deg(:,1)), median(sigmaError_deg(:,2)), iqr(sigmaError_deg(:,2)));
fprintf('Relative error (median(IQR)): %7.2f%% (%.2f%%)    %7.2f%% (%.2f%%)\n', ...
    median(sigmaErrorRel(:,1)), iqr(sigmaErrorRel(:,1)), median(sigmaErrorRel(:,2)), iqr(sigmaErrorRel(:,2)));
fprintf('================================================================================\n');