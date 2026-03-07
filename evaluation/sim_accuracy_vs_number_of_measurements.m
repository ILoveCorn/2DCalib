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
    '3 - Proposed (GPU)'
    '4 - TRO-2016'
    '5 - RCIM-2020'
    '6 - IROS-2015'
    '7 - MEAS-2023'};

% select methods to test
testedMethods = [1 2 4 5 6 7];
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
% number of Euler angles
mAlpha_ = round(sqrt(m/mt));
mBeta_ = mAlpha_;

% repeated times
nTest = 100;

% intial value deviation
dt = 5;                 % hand-eye position (mm)
dTheta = deg2rad(5);    % hand-eye orientation (rad)

%% Method settings
% max iterations for proposed method
iImax = 30;
iOmax = 50;

% convergence tolerance
tol = 1e-5;

%% Simulation
% result
ePos = zeros(nTest, nm, nMethod);
eRot = zeros(nTest, nm, nMethod);
Time = zeros(nTest, nm, nMethod);

% loop count for method 2 (plane_Jnt_GH_debug)
IIout = zeros(nTest, nm);   % max inner loop iterations
IOout = zeros(nTest, nm);   % outer loop iterations

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
                    [tO2He,RO2He,ne,de,tH2We,RH2We,sigmae,time] = plane_Jnt_GH_gpu(tO2H0,RO2H0,nP0,dP0,sigma0,tH2Wn,RH2Wn,pOn,vOn,iImax,iOmax,tol);
                case 4
                    % [tH2Wn,RH2Wn,pOn,vOn,scanO] = simPlane(pP,nP,lP,xMax,zMax,tO2H,RO2H,Sigmat,SigmaR,sigmaO,alphaMax,betaMax,mAlphai,mBetai);
                    [tO2He,RO2He,ne,time] = plane_TRO2016(tO2H0,RO2H0,tH2Wn,RH2Wn,vOn,scanO);
                case 5
                    % [tH2Wn,RH2Wn,pOn,vOn,scanO] = simPlane(pP,nP,lP,xMax,zMax,tO2H,RO2H,Sigmat,SigmaR,sigmaO,alphaMax,betaMax,mAlphai,mBetai);
                    [tO2He,RO2He,ne,time] = plane_RCIM2020(tO2H0,RO2H0,tH2Wn,RH2Wn,pOn,tol);
                case 6
                    [tH2Wn,RH2Wn,pOn] = simIROS2015(nPlane,pPs,nPs,lPs,xMax,zMax,tO2H,RO2H,Sigmat,SigmaR,sigmaO,alphaMax,betaMax,mAlphai,mBetai);
                    [tO2He,RO2He,ne,time] = plane_IROS2015(tO2H0,RO2H0,tH2Wn,RH2Wn,pOn,tol);
                case 7
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

%% Time-number of measurements graph
% mean (nm x nMethod)
TimeMean = reshape(mean(Time, 1), nm, nMethod);

% standard deviation
TimeStd = reshape(std(Time, 1, 1), nm, nMethod);

% legend labels
legendLabels = {
    'Xu'
    'Proposed (CPU)'
    'Proposed (GPU)'
    'Kang'
    'Sharifzadeh'
    'Carlson'
    'Tang'};

% line markers for each method
markers = {'o','*','p','s','d','^','v'};

% figure setting
fig = figure('Units','centimeters', 'Position',[2 2 8.8 8.8], 'Color','w');
ax  = axes(fig);
hold(ax,'on');
grid(ax,'on');
box(ax,'on');

% plot each method (in a different order, Proposed methods last to stay on top)
for i = 1:length(testedMethods)
    plot(ax, m, TimeMean(:,i), '-', ...
        'LineWidth', 1.25, ...
        'Marker', markers{testedMethods(i)}, ...
        'MarkerFaceColor', 'w', ...
        'DisplayName', legendLabels{testedMethods(i)});
end

% axis tick
ax.YScale = 'log';
ax.YMinorGrid = 'on';
ax.YMinorTick = 'on';
xlim(ax, [m(1) m(end)]);
xticks(ax, m);
ax.TickLabelInterpreter = 'latex';
ax.FontSize = 9;

% axis label
xlabel(ax, 'No. of measurements', 'Interpreter', 'latex', 'FontSize', 10);
ylabel(ax, 'Runtime $[\mathrm{s}]$', 'Interpreter', 'latex', 'FontSize', 10);

% legend
leg = legend(ax, 'Location','best', 'Interpreter','latex', 'FontSize', 9);

%% Save result
% comments = {};
% 
% % Clear figure handles before saving
% clearvars fig ax leg t
% 
% dateTimeSuffix = string(datetime('now'), 'yyMMdd_HHmm');
% resultFilename = "result/sim_accuracy_vs_number_of_measurements_" + dateTimeSuffix + ".mat";
% save(resultFilename)