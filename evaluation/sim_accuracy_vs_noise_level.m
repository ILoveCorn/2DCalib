% sim_accuracy_vs_noise_level.m
%   test calibration accuracy under different noise settings

clearvars
close all
clc

rng('default')

%% Methods
allMethods = {
    '1 - RCIM-2022'
    '2 - Proposed'
    '3 - TRO-2016'
    '4 - RCIM-2020'
    '5 - IROS-2015'
    '6 - MEAS-2023'};

% select methods to test
testedMethods = [1 2 3 4 5 6];
nMethod = length(testedMethods);

%% Simulation settings
% sensor field-of-view
xMax = 5;
zMax = 2;

% variance factor
sigmat = [0.01 0.1 0.5 1];  % position (mm)
sigmaRd = [0.01 0.1 0.5 1]; % orientation (degree)
sigmaR = deg2rad(sigmaRd);  % orientation (rad)
sigmaO = 0;                 % image (mm)
nVF = length(sigmat);

% covariance
nCov = 2;
covType = {'isotropic', 'anisotropic'};
Sigmat = zeros(3,3,nCov);
SigmaR = zeros(3,3,nCov);
% isotropic
Sigmat(:,:,1) = eye(3);
SigmaR(:,:,1) = eye(3);
% anisotropic
Sigmat(:,:,2) = diag([0.5 0.5 2]);
SigmaR(:,:,2) = diag([0.5 1 1.5]);

% OCT orientation range
alphaMax = deg2rad(45);
betaMax = deg2rad(45);

% number of Euler angles
mAlpha = 10;
mBeta = mAlpha;
% number of measurements
m = mAlpha*mBeta;

% for RCIM-2022, MEAS-2023
% number of positions per orientation
mt = 3;
% number if Euler angles
mAlpha_ = round(sqrt(m/mt));
mBeta_ = mAlpha_;

% repeated times
nTest = 100;

% intial value deviation
dt = 5;                 % hand-eye position (mm)
dTheta = deg2rad(5);    % hand-eye orientation (rad)

%% Method settings
% max iterations for proposed method
iImax = 100;
iOmax = 20;

% convergence tolerance
tol = 1e-5;

%% Simulation
% result
ePos = zeros(nTest, nVF, nCov, nMethod);
eRot = zeros(nTest, nVF, nCov, nMethod);
Time = zeros(nTest, nVF, nCov, nMethod);

for iCov = 1:nCov
    fprintf('Covariance type: %s\n', covType{iCov})

    for iVF = 1:nVF
        % noise setting
        sigmati = sigmat(iVF);
        sigmaRdi = sigmaRd(iVF);
        sigmaRi = sigmaR(iVF);

        Sigmati = sigmati^2 * Sigmat(:,:,iCov);
        SigmaRi = sigmaRi^2 * SigmaR(:,:,iCov);

        fprintf('\tNoise setting: %.2f mm, %.2f °\n', sigmati, sigmaRdi)

        for iTest = 1:nTest
            fprintf('\t\tRepeat test: %d / %d\n', iTest, nTest)

            % random hand-eye transformation
            tO2H = 50*(2*rand(3,1)-1);
            RO2H = ExpSO3(randSO3());

            % random target setting
            % line
            pL = zeros(3,1);    % starting point
            vL = rand(3,1);
            vL = vL / norm(vL); % direction
            lL = 50;            % length
            % plane
            pP = pL;    % center
            nP = vL;    % normal
            lP = 50;    % side length
            % tri-plane (for IROS-2015)
            nPlane = 3;
            pPs = repmat(pL,[1 nPlane]);    % center
            nPs = ExpSO3(randSO3());        % normal
            lPs = repmat(lP,[1 nPlane]);    % side length
            % sphere
            pS = pP;    % center
            rS = 7.5;   % radius

            % random initial value
            tO2H0 = tO2H + dt*(2*rand(3,1)-1);
            RO2H0 = RO2H*axang2rotm([rand(1,3) dTheta*(2*rand-1)]);
            nP0 = nP + 0.05*(2*rand(3,1)-1);
            nP0 = nP0 / norm(nP0);
            dP0 = pP'*nP0;

            % estimated pose covariance
            kMin = 0.75;
            kMax = 1.25;
            kt = kMin + (kMax-kMin)*rand;
            kR = kMin + (kMax-kMin)*rand;
            sigma0 = [kt*sigmati kR*sigmaRi];

            % test different method
            for iMethod = 1:nMethod
                fprintf('\t\t\tTesting method-%d\n', testedMethods(iMethod))

                switch testedMethods(iMethod)
                    case 1
                        [tH2Wn,RH2Wn,pOn] = simRCIM2022(pL,vL,lL,xMax,zMax,tO2H,RO2H,Sigmati,SigmaRi,sigmaO,alphaMax,betaMax,mAlpha_,mBeta_,mt);
                        [tO2He,RO2He,time] = line_RCIM2022(tH2Wn,RH2Wn,pOn);
                    case 2
                        [tH2Wn,RH2Wn,pOn,vOn,scanO] = simPlane(pP,nP,lP,xMax,zMax,tO2H,RO2H,Sigmati,SigmaRi,sigmaO,alphaMax,betaMax,mAlpha,mBeta);
                        [tO2He,RO2He,ne,de,tH2We,RH2We,sigmae,time] = plane_Jnt_GH(tO2H0,RO2H0,nP0,dP0,sigma0,tH2Wn,RH2Wn,pOn,vOn,iImax,iOmax,tol);
                    case 3
                        % [tH2Wn,RH2Wn,pOn,vOn,scanO] = simPlane(pP,nP,lP,xMax,zMax,tO2H,RO2H,Sigmati,SigmaRi,sigmaO,alphaMax,betaMax,mAlpha,mBeta);
                        [tO2He,RO2He,ne,time] = plane_TRO2016(tO2H0,RO2H0,tH2Wn,RH2Wn,vOn,scanO);
                    case 4
                        % [tH2Wn,RH2Wn,pOn,vOn,scanO] = simPlane(pP,nP,lP,xMax,zMax,tO2H,RO2H,Sigmati,SigmaRi,sigmaO,alphaMax,betaMax,mAlpha,mBeta);
                        [tO2He,RO2He,ne,time] = plane_RCIM2020(tO2H0,RO2H0,tH2Wn,RH2Wn,pOn,tol);
                    case 5
                        [tH2Wn,RH2Wn,pOn] = simIROS2015(nPlane,pPs,nPs,lPs,xMax,zMax,tO2H,RO2H,Sigmati,SigmaRi,sigmaO,alphaMax,betaMax,mAlpha,mBeta);
                        [tO2He,RO2He,ne,time] = plane_IROS2015(tO2H0,RO2H0,tH2Wn,RH2Wn,pOn,tol);
                    case 6
                        [tH2Wn,RH2Wn,pcOe] = simMEAS2023(pS,rS,xMax,zMax,tO2H,RO2H,Sigmati,SigmaRi,sigmaO,alphaMax,betaMax,mAlpha_,mBeta_,mt);
                        [tO2He,RO2He,time] = sphere_MEAS2023(tH2Wn,RH2Wn,pcOe);
                    otherwise
                        error('Method index out of range!')
                end

                % save result
                [ePos(iTest, iVF, iCov, iMethod), eRot(iTest, iVF, iCov, iMethod)] = calibrationError(tO2H,RO2H,tO2He,RO2He);
                Time(iTest, iVF, iCov, iMethod) = time;
            end
        end
    end
end

%% Plot result
% mean (nVF x nCov x nMethod)
ePosMean = reshape(mean(ePos, 1), nVF, nCov, nMethod);
eRotMean = reshape(mean(eRot, 1), nVF, nCov, nMethod);
TimeMean = reshape(mean(Time, 1), nVF, nCov, nMethod);

% standard deviation
ePosStd = reshape(std(ePos, 1, 1), nVF, nCov, nMethod);
eRotStd = reshape(std(eRot, 1, 1), nVF, nCov, nMethod);
TimeStd = reshape(std(Time, 1, 1), nVF, nCov, nMethod);

for iCov = 1:nCov
    %%% calibration error - noise %%%
    figure
    titleStr = sprintf('Covariance type: %s', covType{iCov});
    sgtitle(titleStr, 'Interpreter', 'latex')

    % ePos
    subplot(1,2,1)
    semilogy(sigmat, squeeze(ePosMean(:,iCov,:)), 'LineWidth', 1.5)
    grid on

    title('Position error vs Position noise', 'Interpreter', 'latex')
    xlabel('$\sigma_{\mathbf{R}}$ $(mm)$', 'Interpreter', 'latex')
    ylabel('$\varepsilon_{\mathbf{t}}$ $(mm)$', 'Interpreter', 'latex')
    legend(allMethods(testedMethods), 'Interpreter', 'latex', 'Location', 'best')
    xlim([sigmat(1) sigmat(end)])

    % eRot
    subplot(1,2,2)
    semilogy(sigmaRd, squeeze(eRotMean(:,iCov,:)), 'LineWidth', 1.5)
    grid on

    title('Orientation error vs Orientation noise', 'Interpreter', 'latex')
    xlabel('$\sigma_{\mathbf{R}}$ $(^{\circ})$', 'Interpreter', 'latex')
    ylabel('$\varepsilon_{\mathbf{R}}$ $(^{\circ})$', 'Interpreter', 'latex')
    legend(allMethods(testedMethods), 'Interpreter', 'latex', 'Location', 'best')
    xlim([sigmaRd(1) sigmaRd(end)])

    %%% time - noise %%%%
    % legend labels
    legendLabels = {
        'Xu'
        'Proposed'
        'Kang'
        'Sharifzadeh'
        'Carlson'
        'Tang'};

    % line markers for each method
    markers = {'o','*','s','d','^','v'};

    % figure setting
    figure('Units','centimeters', 'Position',[2 2 8.8 8.8], 'Color','w');
    ax  = gca;
    hold(ax,'on');
    grid(ax,'on');
    box(ax,'on');

    % plot each method (in a different order)
    for iMethod = [2 5 3 4 1 6]
        semilogy(ax, 1:nVF, TimeMean(:,iCov,iMethod), '-', ...
            'LineWidth', 1.25, ...
            'Marker', markers{iMethod}, ...
            'MarkerFaceColor', 'w', ...
            'DisplayName', legendLabels{iMethod});
    end

    % axis tick
    ax.YScale = 'log';
    ax.YMinorGrid = 'on';
    ax.YMinorTick = 'on';
    xlim(ax, [1 nVF]);
    xticks(ax, 1:nVF);
    xticklabels(ax, {'\#1' '\#2' '\#3' '\#4'});
    ax.TickLabelInterpreter = 'latex';
    ax.FontSize = 9;

    % title
    title(ax, titleStr)

    % axis label
    xlabel(ax, 'Noise setting', 'Interpreter', 'latex', 'FontSize', 10);
    ylabel(ax, 'Computation time $(\mathrm{s})$', 'Interpreter', 'latex', 'FontSize', 10);

    % legend
    leg = legend(ax, 'Location','best', 'Interpreter','latex', 'FontSize', 8);

end

%% Print result for manuscript
% ePos
fprintf('\nPosition calibration error\n')
for iMethod = 1:nMethod
    fprintf('%s:\n',allMethods{testedMethods(iMethod)})
    for iVF = 1:nVF
        fprintf('& \\err{%.3f}{%.3f} ',ePosMean(iVF,1,iMethod),ePosStd(iVF,1,iMethod))
        fprintf('& \\err{%.3f}{%.3f} ',ePosMean(iVF,2,iMethod),ePosStd(iVF,2,iMethod))
    end
    fprintf('\n')
end

% eRot
fprintf('\nOrientation calibration error\n')
for iMethod = 1:nMethod
    fprintf('%s:\n',allMethods{testedMethods(iMethod)})
    for iVF = 1:nVF
        fprintf('& \\err{%.3f}{%.3f} ',eRotMean(iVF,1,iMethod),eRotStd(iVF,1,iMethod))
        fprintf('& \\err{%.3f}{%.3f} ',eRotMean(iVF,2,iMethod),eRotStd(iVF,2,iMethod))
    end
    fprintf('\n')
end

% Time
fprintf('\nComputation time\n')
for iMethod = 1:nMethod
    fprintf('%s:\n',allMethods{testedMethods(iMethod)})
    for iVF = 1:nVF
        fprintf('& \\err{%.3f}{%.3f} ',1000*TimeMean(iVF,1,iMethod),1000*TimeStd(iVF,1,iMethod))
        fprintf('& \\err{%.3f}{%.3f} ',1000*TimeMean(iVF,2,iMethod),1000*TimeStd(iVF,2,iMethod))
    end
    fprintf('\n')
end

%% Accuracy improvement
ePosImprv = zeros(nVF,nCov);
eRotImprv = zeros(nVF,nCov);

for iVF = 1:nVF
    for iCov = 1:nCov
        % ePos
        ePosMin2 = mink(ePosMean(iVF,iCov,:),2);
        ePosImprv(iVF,iCov) = 1 - ePosMin2(:,:,1)/ePosMin2(:,:,2);
        % eRot
        eRotMin2 = mink(eRotMean(iVF,iCov,:),2);
        eRotImprv(iVF,iCov) = 1 - eRotMin2(:,:,1)/eRotMin2(:,:,2);
    end
end

fprintf('Average improvement of position error: %.1f %%\n', 100*mean(ePosImprv,'all'))
fprintf('Average improvement of orientation error: %.1f %%\n', 100*mean(eRotImprv,'all'))

%% Save result
comments = {};

dateTimeSuffix = string(datetime('now'), 'yyMMdd_HHmm');
resultFilename = "result/sim_accuracy_vs_noise_level_" + dateTimeSuffix + ".mat";
save(resultFilename)