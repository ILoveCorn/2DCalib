% sim_accuracy_vs_number_of_measurements.m
%   test calibration accuracy under different number of measurements

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
sigmat = 0.5;               % position (mm)
sigmaRd = 0.5;              % orientation (degree)
sigmaR = deg2rad(sigmaRd);  % orientation (rad)
sigmaO = 0;                 % image (mm)

% covariance
Sigmat = sigmat^2 * eye(3);
SigmaR = sigmaR^2 * eye(3);

% OCT orientation range
alphaMax = deg2rad(45);
betaMax = deg2rad(45);

% number of Euler angles
mAlpha = [6 8 10 14 20 28 32];
mBeta = mAlpha;
% number of measurements
m = mAlpha.*mBeta;
nm = length(m);

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
ePos = zeros(nTest, nm, nMethod);
eRot = zeros(nTest, nm, nMethod);
Time = zeros(nTest, nm, nMethod);

for im = 1:nm
    % number of measurements
    mAlphai = mAlpha(im);
    mBetai = mBeta(im);
    mAlpha_i = mAlpha_(im);
    mBeta_i = mBeta_(im);

    fprintf('Number of measurements: %d\n', m(im))

    for iTest = 1:nTest
        fprintf('\tRepeat test: %d / %d\n', iTest, nTest)

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
        sigma0 = [kt*sigmat kR*sigmaR];

        % test different method
        for iMethod = 1:nMethod
            fprintf('\t\tTesting method-%d\n', testedMethods(iMethod))

            switch testedMethods(iMethod)
                case 1
                    [tH2Wn,RH2Wn,pOn] = simRCIM2022(pL,vL,lL,xMax,zMax,tO2H,RO2H,Sigmat,SigmaR,sigmaO,alphaMax,betaMax,mAlpha_i,mBeta_i,mt);
                    [tO2He,RO2He,time] = line_RCIM2022(tH2Wn,RH2Wn,pOn);
                case 2
                    [tH2Wn,RH2Wn,pOn,vOn,scanO] = simPlane(pP,nP,lP,xMax,zMax,tO2H,RO2H,Sigmat,SigmaR,sigmaO,alphaMax,betaMax,mAlphai,mBetai);
                    [tO2He,RO2He,ne,de,tH2We,RH2We,sigmae,time] = plane_Jnt_GH(tO2H0,RO2H0,nP0,dP0,sigma0,tH2Wn,RH2Wn,pOn,vOn,iImax,iOmax,tol);
                case 3
                    % [tH2Wn,RH2Wn,pOn,vOn,scanO] = simPlane(pP,nP,lP,xMax,zMax,tO2H,RO2H,Sigmat,SigmaR,sigmaO,alphaMax,betaMax,mAlphai,mBetai);
                    [tO2He,RO2He,ne,time] = plane_TRO2016(tO2H0,RO2H0,tH2Wn,RH2Wn,vOn,scanO);
                case 4
                    % [tH2Wn,RH2Wn,pOn,vOn,scanO] = simPlane(pP,nP,lP,xMax,zMax,tO2H,RO2H,Sigmat,SigmaR,sigmaO,alphaMax,betaMax,mAlphai,mBetai);
                    [tO2He,RO2He,ne,time] = plane_RCIM2020(tO2H0,RO2H0,tH2Wn,RH2Wn,pOn,tol);
                case 5
                    [tH2Wn,RH2Wn,pOn] = simIROS2015(nPlane,pPs,nPs,lPs,xMax,zMax,tO2H,RO2H,Sigmat,SigmaR,sigmaO,alphaMax,betaMax,mAlphai,mBetai);
                    [tO2He,RO2He,ne,time] = plane_IROS2015(tO2H0,RO2H0,tH2Wn,RH2Wn,pOn,tol);
                case 6
                    [tH2Wn,RH2Wn,pcOe] = simMEAS2023(pS,rS,xMax,zMax,tO2H,RO2H,Sigmat,SigmaR,sigmaO,alphaMax,betaMax,mAlpha_i,mBeta_i,mt);
                    [tO2He,RO2He,time] = sphere_MEAS2023(tH2Wn,RH2Wn,pcOe);
                otherwise
                    error('Method index out of range!')
            end

            % save result
            [ePos(iTest, im, iMethod), eRot(iTest, im, iMethod)] = calibrationError(tO2H,RO2H,tO2He,RO2He);
            Time(iTest, im, iMethod) = time;
        end
    end
end

%% Plot result
% mean (nm x nMethod)
ePosMean = reshape(mean(ePos, 1), nm, nMethod);
eRotMean = reshape(mean(eRot, 1), nm, nMethod);
TimeMean = reshape(mean(Time, 1), nm, nMethod);

% standard deviation
ePosStd = reshape(std(ePos, 1, 1), nm, nMethod);
eRotStd = reshape(std(eRot, 1, 1), nm, nMethod);
TimeStd = reshape(std(Time, 1, 1), nm, nMethod);

figure

% ePos
subplot(1,2,1)
semilogy(m, ePosMean, 'LineWidth', 1.5)
grid on

title('$e_{\rm Pos}$ vs. Number of measurements', 'Interpreter', 'latex')
xlabel('$m$', 'Interpreter', 'latex')
ylabel('Position error $(mm)$', 'Interpreter', 'latex')
legend(allMethods(testedMethods), 'Interpreter', 'latex', 'Location', 'best')
xlim([m(1) m(end)])

% eRot
subplot(1,2,2)
semilogy(m, eRotMean, 'LineWidth', 1.5)
grid on

title('$e_{\rm Rot}$ vs. Number of measurements', 'Interpreter', 'latex')
xlabel('$m$', 'Interpreter', 'latex')
ylabel('Orientation error $(^{\circ})$', 'Interpreter', 'latex')
legend(allMethods(testedMethods), 'Interpreter', 'latex', 'Location', 'best')
xlim([m(1) m(end)])

% Time
figure
semilogy(m, TimeMean, 'LineWidth', 1.5)
grid on

title('Computation time vs. Number of measurements', 'Interpreter', 'latex')
xlabel('Number of meansurements', 'Interpreter', 'latex')
ylabel('Computation time $(\mathrm{s})$', 'Interpreter', 'latex')
legend(allMethods(testedMethods), 'Interpreter', 'latex', 'Location', 'best')
xlim([m(1) m(end)])

%% Time-number of measurements graph
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
fig = figure('Units','centimeters', 'Position',[2 2 4.4 12], 'Color','w');
ax  = axes(fig);
hold(ax,'on');
grid(ax,'on');
box(ax,'on');

% plot each method (in a different order)
for iMethod = [2 5 3 4 1 6]
    semilogy(ax, m(3:end), TimeMean(3:end,iMethod), '-', ...
        'LineWidth', 1.25, ...
        'Marker', markers{iMethod}, ...
        'MarkerFaceColor', 'w', ...
        'DisplayName', legendLabels{iMethod});
end

% axis tick
ax.YScale = 'log';
ax.YMinorGrid = 'on';
ax.YMinorTick = 'on';
xlim(ax, [m(3) m(end)]);
xticks(ax, m(3:end));
% xticklabels(ax, {'\#1' '\#2' '\#3' '\#4'});
ax.TickLabelInterpreter = 'latex';
ax.FontSize = 9;

% axis label
xlabel(ax, 'No. of measurements', 'Interpreter', 'latex', 'FontSize', 10);
ylabel(ax, 'Computation time $(\mathrm{s})$', 'Interpreter', 'latex', 'FontSize', 10);

% legend
leg = legend(ax, 'Location','northoutside', 'Interpreter','latex', 'FontSize', 9);

%% Print result for manuscript
% ePos
fprintf('\nPosition calibration error\n')
for iMethod = 1:nMethod
    fprintf('%s:\n',allMethods{testedMethods(iMethod)})
    for im = 1:nm
        fprintf('& \\err{%.3f}{%.3f} ',ePosMean(im,iMethod),ePosStd(im,iMethod))
    end
    fprintf('\n')
end

% eRot
fprintf('\nOrientation calibration error\n')
for iMethod = 1:nMethod
    fprintf('%s:\n',allMethods{testedMethods(iMethod)})
    for im = 1:nm
        fprintf('& \\err{%.3f}{%.3f} ',eRotMean(im,iMethod),eRotStd(im,iMethod))
    end
    fprintf('\n')
end

% Time
fprintf('\nComputation time\n')
for iMethod = 1:nMethod
    fprintf('%s:\n',allMethods{testedMethods(iMethod)})
    for im = 1:nm
        fprintf('& \\err{%.3f}{%.3f} ',1000*TimeMean(im,iMethod),1000*TimeStd(im,iMethod))
    end
    fprintf('\n')
end

%% Save result
comments = {};

dateTimeSuffix = string(datetime('now'), 'yyMMdd_HHmm');
resultFilename = "result/sim_accuracy_vs_number_of_measurements_" + dateTimeSuffix + ".mat";
save(resultFilename)