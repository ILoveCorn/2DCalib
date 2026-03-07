% sim_accuracy_vs_image_noise.m
%   test calibration accuracy of plane_Jnt_GH under different
%   OCT image noise / distortion conditions

clearvars
close all
clc

rng('default')

%% Conditions
conditionNames = {
    'Ideal'
    'Speckle Noise'
    'Scanning Distortion'};
nCondition = length(conditionNames);

%% Simulation settings
% sensor field-of-view
xMax = 5;
zMax = 2;

% pose noise level (isotropic, 0.5 mm / 0.5 deg)
sigmat = 0.5;               % position (mm)
sigmaRd = 0.5;              % orientation (degree)
sigmaR = deg2rad(sigmaRd);  % orientation (rad)
sigmaO = 0;                 % image noise for simPlane (ideal)

% covariance (isotropic)
Sigmat = sigmat^2 * eye(3);
SigmaR = sigmaR^2 * eye(3);

% OCT orientation range
alphaMax = deg2rad(45);
betaMax = deg2rad(45);

% number of Euler angles
mAlpha = 10;
mBeta = mAlpha;
% number of measurements
m = mAlpha * mBeta;

% repeated times
nTest = 100;

% initial value deviation
dt = 5;                 % hand-eye position (mm)
dTheta = deg2rad(5);    % hand-eye orientation (rad)

%% Image noise / distortion parameters
% Speckle Noise
speckleMuZ = 0.5513e-3;                 % mean of depth noise (mm)
speckleSigmaZ = 0.5263e-3;              % std of depth noise (mm)
speckleMuAngle = deg2rad(0.0001);       % mean of angular perturbation (rad)
speckleSigmaAngle = deg2rad(0.0046);    % std of angular perturbation (rad)

% Scanning Distortion
distortMuZ = -0.3942e-3;                % mean of depth distortion (mm)
distortSigmaZ = 0.6997e-3;              % std of depth distortion (mm)
distortMuAngle = deg2rad(-0.0030);      % mean of angular distortion (rad)
distortSigmaAngle = deg2rad(0.0083);    % std of angular distortion (rad)

%% Method settings
% max iterations
iImax = 30;
iOmax = 50;

% convergence tolerance
tol = 1e-5;

%% Simulation
ePos = zeros(nTest, nCondition);
eRot = zeros(nTest, nCondition);

for iTest = 1:nTest
    fprintf('Repeat test: %d / %d\n', iTest, nTest)

    % random hand-eye transformation
    tO2H = 50*(2*rand(3,1)-1);
    RO2H = ExpSO3(randSO3());

    % plane parameters
    pP = zeros(3,1);    % center point
    nP = rand(3,1);
    nP = nP / norm(nP); % normal
    lP = 50;            % side length

    % initial value
    tO2H0 = tO2H + dt*(2*rand(3,1)-1);
    RO2H0 = RO2H*axang2rotm([rand(1,3) dTheta*(2*rand-1)]);
    nP0 = nP + 0.05*(2*rand(3,1)-1);
    nP0 = nP0 / norm(nP0);
    dP0 = pP'*nP0;

    % initial variance factor
    kMin = 0.75;
    kMax = 1.25;
    kt = kMin + (kMax-kMin)*rand;
    kR = kMin + (kMax-kMin)*rand;
    sigma0 = [kt*sigmat kR*sigmaR];

    % generate ideal data
    [tH2Wn,RH2Wn,pOn,vOn,scanO] = simPlane(pP,nP,lP,xMax,zMax,tO2H,RO2H,Sigmat,SigmaR,sigmaO,alphaMax,betaMax,mAlpha,mBeta);

    for iCond = 1:nCondition
        fprintf('\tCondition %d: %s\n', iCond, conditionNames{iCond})

        % copy ideal data
        pOni = pOn;
        vOni = vOn;

        switch iCond
            case 1  % Ideal
                % no modification

            case 2  % Speckle Noise
                % add Gaussian noise to pOn z-component
                pOni(3,1,:) = pOn(3,1,:) + speckleMuZ + speckleSigmaZ*randn(1,1,m);
                % add angular perturbation to vOn
                for j = 1:m
                    theta0 = atan2(vOn(3,1,j), vOn(1,1,j));
                    thetaN = theta0 + speckleMuAngle + speckleSigmaAngle*randn;
                    vOni(:,1,j) = [cos(thetaN) 0 sin(thetaN)]';
                end

            case 3  % Scanning Distortion
                % add Gaussian distortion to pOn z-component
                pOni(3,1,:) = pOn(3,1,:) + distortMuZ + distortSigmaZ*randn(1,1,m);
                % add angular distortion to vOn
                for j = 1:m
                    theta0 = atan2(vOn(3,1,j), vOn(1,1,j));
                    thetaN = theta0 + distortMuAngle + distortSigmaAngle*randn;
                    vOni(:,1,j) = [cos(thetaN) 0 sin(thetaN)]';
                end
        end

        % calibration
        [tO2He,RO2He,~,~,~,~,~,~] = plane_Jnt_GH(tO2H0,RO2H0,nP0,dP0,sigma0,tH2Wn,RH2Wn,pOni,vOni,iImax,iOmax,tol);
        [ePos(iTest,iCond),eRot(iTest,iCond)] = calibrationError(tO2H,RO2H,tO2He,RO2He);
    end
end

%% Statistics - Median (IQR)
ePosMedian = median(ePos);
eRotMedian = median(eRot);
ePosIQR = iqr(ePos);
eRotIQR = iqr(eRot);

% relative error increase (%)
ePosIncrease = (ePosMedian - ePosMedian(1)) / ePosMedian(1) * 100;
eRotIncrease = (eRotMedian - eRotMedian(1)) / eRotMedian(1) * 100;

% print statistics
fprintf('\n=== Calibration Error: Median (IQR) and Relative Increase ===\n')
fprintf('%-30s\t%20s\t%20s\t%12s\t%12s\n', ...
    'Condition', 'e_t (μm)', 'e_R (deg)', 'Δe_t (%)', 'Δe_R (%)')
fprintf('%s\n', repmat('-', 1, 100))
for iCond = 1:nCondition
    if iCond == 1  % Ideal condition
        fprintf('%-30s\t%8.4f (%7.4f)\t%8.4f (%7.4f)\t%12s\t%12s\n', ...
            conditionNames{iCond}, ...
            1000*ePosMedian(iCond), 1000*ePosIQR(iCond), ...
            eRotMedian(iCond), eRotIQR(iCond), ...
            '-', '-')
    else
        fprintf('%-30s\t%8.4f (%7.4f)\t%8.4f (%7.4f)\t%11.2f\t%11.2f\n', ...
            conditionNames{iCond}, ...
            1000*ePosMedian(iCond), 1000*ePosIQR(iCond), ...
            eRotMedian(iCond), eRotIQR(iCond), ...
            ePosIncrease(iCond), eRotIncrease(iCond))
    end
end
