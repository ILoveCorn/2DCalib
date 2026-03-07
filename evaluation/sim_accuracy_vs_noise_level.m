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
    '3 - Proposed (Alt)'
    '4 - Proposed (Decoupled)'
    '5 - TRO-2016'
    '6 - RCIM-2020'
    '7 - IROS-2015'
    '8 - MEAS-2023'};

% select methods to test
testedMethods = [1 2 5 6 7 8];
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
% alphaMax = 0;
betaMax = deg2rad(45);
% betaMax = 0;

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
% mAlpha_ = mAlpha;
mBeta_ = mAlpha_;

% repeated times
nTest = 100;

% intial value deviation
dt = 5;                 % hand-eye position (mm)
dTheta = deg2rad(5);    % hand-eye orientation (rad)

% plane fitting test
nPlaneFit = 50;         % number of plane fitting tests per calibration

%% Method settings
% max iterations for proposed method
iImax = 30;
iOmax = 50;

% convergence tolerance
tol = 1e-5;

%% Simulation
% result
ePos = zeros(nTest, nVF, nCov, nMethod);
eRot = zeros(nTest, nVF, nCov, nMethod);
Time = zeros(nTest, nVF, nCov, nMethod);

% plane fitting residuals
planeFitRes = zeros(nTest, nVF, nCov, nMethod, nPlaneFit);

% loop count for method 2 (plane_Jnt_GH_debug)
IIout = zeros(nTest, nVF, nCov);   % max inner loop iterations
IOout = zeros(nTest, nVF, nCov);   % outer loop iterations

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
                        [tO2He,RO2He,ne,de,tH2We,RH2We,sigmae,time] = plane_Alt_GH_debug(tO2H0,RO2H0,nP0,dP0,sigma0,tH2Wn,RH2Wn,pOn,vOn,iImax,iOmax,tol);
                    case 4
                        % [tH2Wn,RH2Wn,pOn,vOn,scanO] = simPlane(pP,nP,lP,xMax,zMax,tO2H,RO2H,Sigmati,SigmaRi,sigmaO,alphaMax,betaMax,mAlpha,mBeta);
                        [tO2He,RO2He,ne,de,tH2We,RH2We,sigmae,time] = plane_Jnt_GH_decoupled(tO2H0,RO2H0,nP0,dP0,sigma0,tH2Wn,RH2Wn,pOn,vOn,iImax,iOmax,tol);
                    case 5
                        % [tH2Wn,RH2Wn,pOn,vOn,scanO] = simPlane(pP,nP,lP,xMax,zMax,tO2H,RO2H,Sigmati,SigmaRi,sigmaO,alphaMax,betaMax,mAlpha,mBeta);
                        [tO2He,RO2He,ne,time] = plane_TRO2016(tO2H0,RO2H0,tH2Wn,RH2Wn,vOn,scanO);
                    case 6
                        % [tH2Wn,RH2Wn,pOn,vOn,scanO] = simPlane(pP,nP,lP,xMax,zMax,tO2H,RO2H,Sigmati,SigmaRi,sigmaO,alphaMax,betaMax,mAlpha,mBeta);
                        [tO2He,RO2He,ne,time] = plane_RCIM2020(tO2H0,RO2H0,tH2Wn,RH2Wn,pOn,tol);
                    case 7
                        [tH2Wn,RH2Wn,pOn] = simIROS2015(nPlane,pPs,nPs,lPs,xMax,zMax,tO2H,RO2H,Sigmati,SigmaRi,sigmaO,alphaMax,betaMax,mAlpha,mBeta);
                        [tO2He,RO2He,ne,time] = plane_IROS2015(tO2H0,RO2H0,tH2Wn,RH2Wn,pOn,tol);
                    case 8
                        [tH2Wn,RH2Wn,pcOe] = simMEAS2023(pS,rS,xMax,zMax,tO2H,RO2H,Sigmati,SigmaRi,sigmaO,alphaMax,betaMax,mAlpha_,mBeta_,mt);
                        [tO2He,RO2He,time] = sphere_MEAS2023(tH2Wn,RH2Wn,pcOe);
                    otherwise
                        error('Method index out of range!')
                end

                % save result
                [ePos(iTest, iVF, iCov, iMethod), eRot(iTest, iVF, iCov, iMethod)] = calibrationError(tO2H,RO2H,tO2He,RO2He);
                Time(iTest, iVF, iCov, iMethod) = time;

                % plane fitting test
                for iPlaneFit = 1:nPlaneFit
                    % generate plane fitting data
                    [tH2WnPF,RH2WnPF,~,~,scanOPF] = simPlane(pP,nP,lP,xMax,zMax,tO2H,RO2H,Sigmati,SigmaRi,sigmaO,alphaMax,betaMax,mAlpha,mBeta);

                    % compute plane fitting residual
                    [~,~,res] = planeFit(tO2He,RO2He,tH2WnPF,RH2WnPF,scanOPF);

                    % save residual
                    planeFitRes(iTest, iVF, iCov, iMethod, iPlaneFit) = res;
                end
            end
        end
    end
end

%% Plot result
% mean (nVF x nCov x nMethod)
ePosMean = reshape(mean(ePos, 1), nVF, nCov, nMethod);
eRotMean = reshape(mean(eRot, 1), nVF, nCov, nMethod);
TimeMean = reshape(mean(Time, 1), nVF, nCov, nMethod);
planeFitResMean = reshape(mean(planeFitRes, [1,5]), nVF, nCov, nMethod);

% standard deviation
ePosStd = reshape(std(ePos, 1, 1), nVF, nCov, nMethod);
eRotStd = reshape(std(eRot, 1, 1), nVF, nCov, nMethod);
TimeStd = reshape(std(Time, 1, 1), nVF, nCov, nMethod);
planeFitResStd = reshape(std(reshape(planeFitRes, nTest*nPlaneFit, nVF, nCov, nMethod), 1, 1), nVF, nCov, nMethod);

%%% calibration error - noise %%%
% Create figure with specified size
figure('Units','centimeters', 'Position',[2 2 17.6 8.8], 'Color','w');

% Create tiled layout
t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% legend labels
legendLabels = {
    'Xu'
    'Proposed'
    'Ablation (Alternating)'
    'Ablation (only $g_2$ for $(\mathbf{R,n})$)'
    'Kang'
    'Sharifzadeh'
    'Carlson'
    'Tang'};

% Line markers for each method
markers = {'o','*','p','h','s','d','^','v'};

% Line styles for covariance types
lineStyles = {'-', '--'};  % solid for isotropic, dashed for anisotropic

% Covariance type suffixes
covSuffix = {', iso.', ', aniso.'};

% ePos - Position error
nexttile
ax1 = gca;
hold(ax1, 'on');
grid(ax1, 'on');
box(ax1, 'on');

for iCov = 1:nCov
    for i = 1:length(testedMethods)
        iMethod = testedMethods(i);
        displayName = [legendLabels{iMethod} covSuffix{iCov} 'noise'];

        if iMethod == 2  % Proposed (Joint) - MATLAB default blue
            plot(ax1, 1:nVF, ePosMean(:,iCov,i), lineStyles{iCov}, ...
                'LineWidth', 1.25, ...
                'Marker', markers{iMethod}, ...
                'MarkerFaceColor', 'w', ...
                'Color', [0 0.4470 0.7410], ...
                'DisplayName', displayName);
        elseif iMethod == 3  % Proposed (Alternating) - MATLAB default orange
            plot(ax1, 1:nVF, ePosMean(:,iCov,i), lineStyles{iCov}, ...
                'LineWidth', 1.25, ...
                'Marker', markers{iMethod}, ...
                'MarkerFaceColor', 'w', ...
                'Color', [0.8500 0.3250 0.0980], ...
                'DisplayName', displayName);
        elseif iMethod == 4  % Proposed (Decoupled) - MATLAB default green
            plot(ax1, 1:nVF, ePosMean(:,iCov,i), lineStyles{iCov}, ...
                'LineWidth', 1.25, ...
                'Marker', markers{iMethod}, ...
                'MarkerFaceColor', 'w', ...
                'Color', [0.4660 0.6740 0.1880], ...
                'DisplayName', displayName);
        else
            plot(ax1, 1:nVF, ePosMean(:,iCov,i), lineStyles{iCov}, ...
                'LineWidth', 1.25, ...
                'Marker', markers{iMethod}, ...
                'MarkerFaceColor', 'w', ...
                'DisplayName', displayName);
        end
    end
end

xlim(ax1, [1 nVF]);
xticks(ax1, 1:nVF);
xticklabels(ax1, {'\#1' '\#2' '\#3' '\#4'});
ax1.TickLabelInterpreter = 'latex';
ax1.FontSize = 8;
ax1.YMinorGrid = 'on';
ax1.YMinorTick = 'on';

title(ax1, 'Position error', 'Interpreter', 'latex', 'FontSize', 9, 'FontWeight', 'bold');
xlabel(ax1, 'Noise setting', 'Interpreter', 'latex', 'FontSize', 9);
ylabel(ax1, '$\varepsilon_{\mathbf{t}}$ $(\mathrm{mm})$', 'Interpreter', 'latex', 'FontSize', 9);
legend(ax1, 'Location', 'best', 'Interpreter', 'latex', 'FontSize', 8);

% eRot - Orientation error
nexttile
ax2 = gca;
hold(ax2, 'on');
grid(ax2, 'on');
box(ax2, 'on');

for iCov = 1:nCov
    for i = 1:length(testedMethods)
        iMethod = testedMethods(i);
        displayName = [legendLabels{iMethod} covSuffix{iCov} ' noise'];

        if iMethod == 2  % Proposed (Joint) - MATLAB default blue
            plot(ax2, 1:nVF, eRotMean(:,iCov,i), lineStyles{iCov}, ...
                'LineWidth', 1.25, ...
                'Marker', markers{iMethod}, ...
                'MarkerFaceColor', 'w', ...
                'Color', [0 0.4470 0.7410], ...
                'DisplayName', displayName);
        elseif iMethod == 3  % Proposed (Alternating) - MATLAB default orange
            plot(ax2, 1:nVF, eRotMean(:,iCov,i), lineStyles{iCov}, ...
                'LineWidth', 1.25, ...
                'Marker', markers{iMethod}, ...
                'MarkerFaceColor', 'w', ...
                'Color', [0.8500 0.3250 0.0980], ...
                'DisplayName', displayName);
        elseif iMethod == 4  % Proposed (Decoupled) - MATLAB default green
            plot(ax2, 1:nVF, eRotMean(:,iCov,i), lineStyles{iCov}, ...
                'LineWidth', 1.25, ...
                'Marker', markers{iMethod}, ...
                'MarkerFaceColor', 'w', ...
                'Color', [0.4660 0.6740 0.1880], ...
                'DisplayName', displayName);
        else
            plot(ax2, 1:nVF, eRotMean(:,iCov,i), lineStyles{iCov}, ...
                'LineWidth', 1.25, ...
                'Marker', markers{iMethod}, ...
                'MarkerFaceColor', 'w', ...
                'DisplayName', displayName);
        end
    end
end

xlim(ax2, [1 nVF]);
xticks(ax2, 1:nVF);
xticklabels(ax2, {'\#1' '\#2' '\#3' '\#4'});
ax2.TickLabelInterpreter = 'latex';
ax2.FontSize = 8;
ax2.YMinorGrid = 'on';
ax2.YMinorTick = 'on';

title(ax2, 'Orientation error', 'Interpreter', 'latex', 'FontSize', 9, 'FontWeight', 'bold');
xlabel(ax2, 'Noise setting', 'Interpreter', 'latex', 'FontSize', 9);
ylabel(ax2, '$\varepsilon_{\mathbf{R}}$ $(^{\circ})$', 'Interpreter', 'latex', 'FontSize', 9);
legend(ax2, 'Location', 'best', 'Interpreter', 'latex', 'FontSize', 8);

%% Median (IQR) statistics
% Calculate median and IQR for position and orientation errors
ePosMedian = zeros(nVF, nCov, nMethod);
eRotMedian = zeros(nVF, nCov, nMethod);
ePosIQR = zeros(nVF, nCov, nMethod);
eRotIQR = zeros(nVF, nCov, nMethod);
TimeMedian = zeros(nVF, nCov, nMethod);
TimeIQR = zeros(nVF, nCov, nMethod);
planeFitResMedian = zeros(nVF, nCov, nMethod);
planeFitResIQR = zeros(nVF, nCov, nMethod);

for iVF = 1:nVF
    for iCov = 1:nCov
        for iMethod = 1:nMethod
            % Position error
            posData = ePos(:, iVF, iCov, iMethod);
            ePosMedian(iVF, iCov, iMethod) = median(posData);
            ePosIQR(iVF, iCov, iMethod) = iqr(posData);

            % Orientation error
            rotData = eRot(:, iVF, iCov, iMethod);
            eRotMedian(iVF, iCov, iMethod) = median(rotData);
            eRotIQR(iVF, iCov, iMethod) = iqr(rotData);

            % Computation time
            timeData = Time(:, iVF, iCov, iMethod);
            TimeMedian(iVF, iCov, iMethod) = median(timeData);
            TimeIQR(iVF, iCov, iMethod) = iqr(timeData);

            % Plane fitting residuals
            pfResData = reshape(planeFitRes(:, iVF, iCov, iMethod, :), [], 1);
            planeFitResMedian(iVF, iCov, iMethod) = median(pfResData);
            planeFitResIQR(iVF, iCov, iMethod) = iqr(pfResData);
        end
    end
end

% Print Median (IQR) for latex manuscript
fprintf('\n=== Median (IQR) Statistics ===\n')

% ePos Median (IQR)
fprintf('\nPosition calibration error - Median (IQR)\n')
for iMethod = 1:nMethod
    fprintf('%s:\n', allMethods{testedMethods(iMethod)})
    for iVF = 1:nVF
        fprintf('& \\mediqr{%.3f}{%.3f} ', ePosMedian(iVF, 1, iMethod), ePosIQR(iVF, 1, iMethod))
        fprintf('& \\mediqr{%.3f}{%.3f} ', ePosMedian(iVF, 2, iMethod), ePosIQR(iVF, 2, iMethod))
    end
    fprintf('\\\\\n')
end

% eRot Median (IQR)
fprintf('\nOrientation calibration error - Median (IQR)\n')
for iMethod = 1:nMethod
    fprintf('%s:\n', allMethods{testedMethods(iMethod)})
    for iVF = 1:nVF
        fprintf('& \\mediqr{%.3f}{%.3f} ', eRotMedian(iVF, 1, iMethod), eRotIQR(iVF, 1, iMethod))
        fprintf('& \\mediqr{%.3f}{%.3f} ', eRotMedian(iVF, 2, iMethod), eRotIQR(iVF, 2, iMethod))
    end
    fprintf('\\\\\n')
end

% Time Median (IQR)
fprintf('\nComputation time - Median (IQR) (ms)\n')
for iMethod = 1:nMethod
    fprintf('%s:\n', allMethods{testedMethods(iMethod)})
    for iVF = 1:nVF
        fprintf('& \\mediqr{%.3f}{%.3f} ', 1000*TimeMedian(iVF, 1, iMethod), 1000*TimeIQR(iVF, 1, iMethod))
        fprintf('& \\mediqr{%.3f}{%.3f} ', 1000*TimeMedian(iVF, 2, iMethod), 1000*TimeIQR(iVF, 2, iMethod))
    end
    fprintf('\\\\\n')
end

% Plane fitting residuals Median (IQR)
fprintf('\nPlane fitting residual - Median (IQR) (mm)\n')
for iMethod = 1:nMethod
    fprintf('%s:\n', allMethods{testedMethods(iMethod)})
    for iVF = 1:nVF
        fprintf('& \\mediqr{%.4f}{%.4f} ', planeFitResMedian(iVF, 1, iMethod), planeFitResIQR(iVF, 1, iMethod))
        fprintf('& \\mediqr{%.4f}{%.4f} ', planeFitResMedian(iVF, 2, iMethod), planeFitResIQR(iVF, 2, iMethod))
    end
    fprintf('\\\\\n')
end

%% Save result
% comments = {};
%
% % Clear figure handles before saving
% clearvars fig ax leg t
%
% dateTimeSuffix = string(datetime('now'), 'yyMMdd_HHmm');
% resultFilename = "result/sim_accuracy_vs_noise_level_" + dateTimeSuffix + ".mat";
% save(resultFilename)