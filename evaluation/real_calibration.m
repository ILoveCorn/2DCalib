% real_calibration.m
%   hand-eye calibration using real data

clearvars
close all
clc

rng('default')

%% Methods
testedMethods = {
    '1 - Proposed'
    '2 - TRO-2016'
    '3 - RCIM-2020'
    '4 - MEAS-2023'
    '5 - Measured'};
nMethod = length(testedMethods);

%% Method settings
% 1. Proposed
% initial hand-eye transformation
load('data\measure_250723_1647.mat','tO2H','RO2H')
tO2H0 = tO2H;
RO2H0 = RO2H;
% variance factor
sigma0 = [0.1 0.01];

% max iterations of inner & outer loop
iImax = 100;
iOmax = 20;

% convergence tolerance
tol = 1e-5;

% 4. MEAS-2023
% number of measurements per orientation
mt = 3;

% sphere target diameter
D = 14.9981;

%% Get calibration data
% plane
planeFiles = dir('data/oct/plane/oct_plane*.yaml');
planeFiles = {planeFiles.name};
nCalibFile = length(planeFiles);

% sphere
sphereFiles = dir('data/oct/sphere/oct_sphere*.yaml');
sphereFiles = {sphereFiles.name};

%% Calibration
% estimation results
alltO2He = zeros(3, nCalibFile, nMethod);
allRO2He = zeros(3, 3, nCalibFile, nMethod);
Time = zeros(nCalibFile, nMethod);

for iCalibFile = 1:nCalibFile
    % display progress
    fprintf('Calibration using file - %d / %d\n', iCalibFile, nCalibFile)

    % 1,2,3: plane-based methods
    [tH2Wn,RH2Wn,pOn,vOn,scanO] = loadPlaneYAML(planeFiles{iCalibFile});

    % initial value of plane parameters
    [n0,d0,~] = planeFit(tO2H0,RO2H0,tH2Wn,RH2Wn,scanO);

    for iMethod = 1:3
        fprintf('\tUsing method-%d\n',iMethod)
        switch iMethod
            case 1
                [tO2He,RO2He,~,~,~,~,~,time] = plane_Jnt_GH(tO2H0,RO2H0,n0,d0,sigma0,tH2Wn,RH2Wn,pOn,vOn,iImax,iOmax,tol);
            case 2
                [tO2He,RO2He,~,time] = plane_TRO2016(tO2H0,RO2H0,tH2Wn,RH2Wn,vOn,scanO);
            case 3
                [tO2He,RO2He,~,time] = plane_RCIM2020(tO2H0,RO2H0,tH2Wn,RH2Wn,pOn,tol);
        end

        alltO2He(:, iCalibFile, iMethod) = tO2He;
        allRO2He(:, :, iCalibFile, iMethod) = RO2He;
        Time(iCalibFile, iMethod) = time;
    end

    % 4: sphere-based method
    iMethod = 4;
    fprintf('\tUsing method-%d\n',iMethod)
    
    [tH2Wn,RH2Wn,pcOe,~] = loadMEAS2023YAML(sphereFiles{iCalibFile},mt,D);
    [tO2He,RO2He,time] = sphere_MEAS2023(tH2Wn,RH2Wn,pcOe);

    alltO2He(:, iCalibFile, iMethod) = tO2He;
    allRO2He(:, :, iCalibFile, iMethod) = RO2He;
    Time(iCalibFile, iMethod) = time;


    % 5: measured
    iMethod = 5;
    alltO2He(:, iCalibFile, iMethod) = tO2H0;
    allRO2He(:, :, iCalibFile, iMethod) = RO2H0;
end

%% Save result
dateTimeSuffix = string(datetime('now'), 'yyMMdd_HHmm');
resultFilename = 'result/real_calibration_' + dateTimeSuffix + '.mat';
save(resultFilename, 'testedMethods', 'planeFiles', 'sphereFiles', 'D', ...
    'alltO2He', 'allRO2He', 'Time', ...
    'sigma0', 'iImax', 'iOmax', 'tol')
