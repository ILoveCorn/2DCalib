% sim_accuracy_vs_initial_deviation.m
%   test convergence rate of the Proposed method under different initial deviations

clearvars
close all
clc

rng('default')

%% Simulation settings
% sensor field-of-view
xMax = 5;
zMax = 2;

% fixed noise setting
sigmat = 0.5;               % position noise (mm)
sigmaRd = 0.5;              % orientation noise (degree)
sigmaR = deg2rad(sigmaRd);  % orientation noise (rad)
sigmaO = 0;                 % image noise (mm)

% noise covariance (isotropic)
Sigmat = sigmat^2 * eye(3);
SigmaR = sigmaR^2 * eye(3);

% OCT orientation range
alphaMax = deg2rad(45);
betaMax = deg2rad(45);

% number of Euler angles
mAlpha = 10;
mBeta = mAlpha;
% number of measurements
m = mAlpha*mBeta;

% initial deviation settings
dThetaDeg = [10, 20, 30, 40, 50];   % orientation deviation (degree)
dt = [1 5 10 15 20];                % position deviation (mm)
dTheta = deg2rad(dThetaDeg);        % orientation deviation (rad)
nDev = length(dt);

% repeated times
nTest = 100;

% convergence threshold: 3x noise level
ePosTh = 3 * sigmat;        % position error threshold (mm)
eRotTh = 3 * sigmaRd;       % orientation error threshold (degree)

%% Method settings
% max iterations for proposed method
iImax = 30;
iOmax = 50;

% convergence tolerance
tol = 1e-5;

%% Simulation
% result
ePos = zeros(nTest, nDev);
eRot = zeros(nTest, nDev);
compTime = zeros(nTest, nDev);  % computation time (s)
iIter = zeros(nTest, nDev);     % inner loop iterations
iOIter = zeros(nTest, nDev);    % outer loop iterations
converged = zeros(nTest, nDev);  % 1 if converged, 0 otherwise

for iDev = 1:nDev
    fprintf('Initial deviation: %.1f°, %.1f mm\n', dThetaDeg(iDev), dt(iDev))

    for iTest = 1:nTest
        fprintf('\tRepeat test: %d / %d\n', iTest, nTest)

        % random hand-eye transformation
        tO2H = 50*(2*rand(3,1)-1);
        RO2H = ExpSO3(randSO3());

        % plane parameters
        pP = zeros(3,1);    % center point
        nP = rand(3,1);
        nP = nP / norm(nP); % normal
        lP = 50;            % side length

        % random initial value with specified deviation
        % position: random direction, specified magnitude
        dtDir = randn(3, 1);
        dtDir = dtDir / norm(dtDir);
        tO2H0 = tO2H + dt(iDev) * dtDir;

        % orientation: random axis, specified angle
        dRAxis = randn(1, 3);
        dRAxis = dRAxis / norm(dRAxis);
        RO2H0 = RO2H * axang2rotm([dRAxis, dTheta(iDev)]);

        % plane initial value
        nP0 = nP + 0.05*(2*rand(3,1)-1);
        nP0 = nP0 / norm(nP0);
        dP0 = pP'*nP0;

        % estimated pose covariance (with random perturbation)
        kMin = 0.75;
        kMax = 1.25;
        kt = kMin + (kMax-kMin)*rand;
        kR = kMin + (kMax-kMin)*rand;
        sigma0 = [kt*sigmat kR*sigmaR];

        % simulated data
        [tH2Wn,RH2Wn,pOn,vOn,~] = simPlane(pP,nP,lP,xMax,zMax,tO2H,RO2H,Sigmat,SigmaR,sigmaO,alphaMax,betaMax,mAlpha,mBeta);

        % Proposed method
        % [tO2He,RO2He,~,~,~,~,~,time] = plane_Jnt_GH(tO2H0,RO2H0,nP0,dP0,sigma0,tH2Wn,RH2Wn,pOn,vOn,iImax,iOmax,tol);
        [tO2He,RO2He,~,~,~,~,~,time,iIout,iOout] = plane_Jnt_GH_debug(tO2H0,RO2H0,nP0,dP0,sigma0,tH2Wn,RH2Wn,pOn,vOn,iImax,iOmax,tol);

        % record computation time and iterations
        compTime(iTest, iDev) = time;
        iIter(iTest, iDev) = iIout;    % inner loop iterations
        iOIter(iTest, iDev) = iOout;   % outer loop iterations

        % compute calibration error
        [ePos(iTest, iDev), eRot(iTest, iDev)] = calibrationError(tO2H,RO2H,tO2He,RO2He);

        % check convergence
        if ePos(iTest, iDev) < ePosTh && eRot(iTest, iDev) < eRotTh
            converged(iTest, iDev) = 1;
        end
    end
end

%% Compute convergence rate
convRate = sum(converged, 1) / nTest * 100;  % percentage

%% Print results
fprintf('\n');
fprintf('================================================================================\n');
fprintf('           Convergence Rate vs Initial Deviation (Proposed Method)\n');
fprintf('================================================================================\n');
fprintf('Noise level: σ_t = %.2f mm, σ_R = %.2f°\n', sigmat, sigmaRd);
fprintf('Convergence threshold: ε_t < %.2f mm, ε_R < %.2f°\n', ePosTh, eRotTh);
fprintf('Number of tests per setting: %d\n', nTest);
fprintf('--------------------------------------------------------------------------------\n');
fprintf('%-20s | %-15s | %-15s\n', 'Initial Deviation', 'Converged/Total', 'Convergence Rate');
fprintf('--------------------------------------------------------------------------------\n');
for iDev = 1:nDev
    fprintf('[%2.0f°, %.1fmm]          | %6d / %-6d  | %6.1f %%\n', ...
        dThetaDeg(iDev), dt(iDev), sum(converged(:,iDev)), nTest, convRate(iDev));
end
fprintf('================================================================================\n');

% Print calibration error, computation time, and iteration statistics
fprintf('\n');
fprintf('================================================================================================================================\n');
fprintf('           Calibration Error, Computation Time, and Iteration Statistics (Median (IQR))\n');
fprintf('================================================================================================================================\n');
fprintf('%-20s | %-20s | %-20s | %-20s | %-13s | %-13s\n', 'Initial Deviation', 'Position Error (mm)', 'Orientation Error (°)', 'Comp. Time (s)', 'Inner Iter.', 'Outer Iter.');
fprintf('--------------------------------------------------------------------------------------------------------------------------------\n');
for iDev = 1:nDev
    fprintf('[%2.0f°, %.1fmm]          | %7.3f (%-7.3f)  | %7.3f (%-7.3f)  | %7.2f (%-7.2f)  | %5.1f (%-4.1f) | %5.1f (%-4.1f)\n', ...
        dThetaDeg(iDev), dt(iDev), ...
        median(ePos(:,iDev)), iqr(ePos(:,iDev)), ...
        median(eRot(:,iDev)), iqr(eRot(:,iDev)), ...
        median(1000*compTime(:,iDev)), iqr(1000*compTime(:,iDev)), ...
        median(iIter(:,iDev)), iqr(iIter(:,iDev)), ...
        median(iOIter(:,iDev)), iqr(iOIter(:,iDev)));
end
fprintf('================================================================================================================================\n');

%% Plot results
fig = figure('Units','centimeters', 'Position',[2 2 12 8], 'Color','w');

% Create x-axis labels
xLabels = cell(1, nDev);
for iDev = 1:nDev
    xLabels{iDev} = sprintf('[%d°, %.1fmm]', dThetaDeg(iDev), dt(iDev));
end

% Plot convergence rate
bar(1:nDev, convRate, 0.6, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 1);
hold on

% Add value labels on bars
for iDev = 1:nDev
    text(iDev, convRate(iDev) + 2, sprintf('%.1f%%', convRate(iDev)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'FontName', 'Times New Roman');
end

% Axis settings
xlim([0.5 nDev+0.5]);
ylim([0 110]);
xticks(1:nDev);
xticklabels(xLabels);
yticks(0:20:100);

% Labels
xlabel('Initial Deviation $[\Delta\theta, \Delta t]$', 'Interpreter', 'latex', 'FontSize', 11);
ylabel('Convergence Rate (\%)', 'Interpreter', 'latex', 'FontSize', 11);
title(sprintf('Convergence Rate vs Initial Deviation ($\\sigma_t=%.1f$ mm, $\\sigma_R=%.1f^\\circ$)', sigmat, sigmaRd), ...
    'Interpreter', 'latex', 'FontSize', 11);

grid on
box on
set(gca, 'FontName', 'Times New Roman', 'FontSize', 9, 'TickLabelInterpreter', 'latex');

hold off

%% Additional plot: Boxplot of calibration errors
fig2 = figure('Units','centimeters', 'Position',[16 2 14 10], 'Color','w');
tiledlayout(1,2,"TileSpacing","compact","Padding","compact")

% Position error
nexttile
boxplot(ePos, 'Labels', xLabels);
ylabel('$\varepsilon_{\mathbf{t}}$ [mm]', 'Interpreter', 'latex', 'FontSize', 10);
xlabel('Initial Deviation', 'Interpreter', 'latex', 'FontSize', 10);
title('Position Error', 'Interpreter', 'latex', 'FontSize', 11);
grid on
box on
set(gca, 'FontName', 'Times New Roman', 'FontSize', 9, 'TickLabelInterpreter', 'latex');

% Orientation error
nexttile
boxplot(eRot, 'Labels', xLabels);
ylabel('$\varepsilon_{\mathbf{R}}$ [$^\circ$]', 'Interpreter', 'latex', 'FontSize', 10);
xlabel('Initial Deviation', 'Interpreter', 'latex', 'FontSize', 10);
title('Orientation Error', 'Interpreter', 'latex', 'FontSize', 11);
grid on
box on
set(gca, 'FontName', 'Times New Roman', 'FontSize', 9, 'TickLabelInterpreter', 'latex');

%% Histogram plot: Distribution of calibration errors
fig3 = figure('Units','centimeters', 'Position',[2 14 18 10], 'Color','w');
tiledlayout(2, nDev, "TileSpacing","compact", "Padding","compact")

% Plot position error histograms (first row)
for iDev = 1:nDev
    nexttile
    histogram(ePos(:,iDev), 'BinMethod', 'auto', 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 0.5);
    xlabel('$\varepsilon_{\mathbf{t}}$ [mm]', 'Interpreter', 'latex', 'FontSize', 9);
    if iDev == 1
        ylabel('Frequency', 'Interpreter', 'latex', 'FontSize', 9);
    end
    title(sprintf('$[%d^\\circ, %.1f$ mm$]$', dThetaDeg(iDev), dt(iDev)), 'Interpreter', 'latex', 'FontSize', 10);
    grid on
    box on
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 8, 'TickLabelInterpreter', 'latex');
end

% Plot orientation error histograms (second row)
for iDev = 1:nDev
    nexttile
    histogram(eRot(:,iDev), 'BinMethod', 'auto', 'FaceColor', [0.9 0.5 0.3], 'EdgeColor', 'k', 'LineWidth', 0.5);
    xlabel('$\varepsilon_{\mathbf{R}}$ [$^\circ$]', 'Interpreter', 'latex', 'FontSize', 9);
    if iDev == 1
        ylabel('Frequency', 'Interpreter', 'latex', 'FontSize', 9);
    end
    grid on
    box on
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 8, 'TickLabelInterpreter', 'latex');
end

%% Save result
% comments = {'Test convergence rate under different initial deviations'};
%
% % Clear figure handles before saving
% clearvars fig ax leg t
%
% dateTimeSuffix = string(datetime('now'), 'yyMMdd_HHmm');


% resultFilename = "result/sim_accuracy_vs_initial_deviation_" + dateTimeSuffix + ".mat";
% save(resultFilename)
