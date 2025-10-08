% real_sphere.m
%   sphere reconstruction experiments

clearvars
close all
clc

%% Load calibration result
load('result/real_calibration_251008_1022.mat', 'testedMethods', 'sphereFiles', 'D', 'alltO2He', 'allRO2He', 'Time')
nMethod = length(testedMethods);
nCalibFile = size(alltO2He, 2);

% get test files
testFiles = dir('data/oct/sphere/oct_sphere*.yaml');
testFiles = {testFiles.name};
nTestFile = length(testFiles);

%% Sphere reconstruction
eSphere = zeros(nTestFile, nCalibFile, nMethod);

for iCalibFile = 1:nCalibFile
    fprintf('Sphere reconstruction using calibration - %d / %d\n', iCalibFile, nCalibFile)

    for iTestFile = 1:nTestFile
        fprintf('\tusing test data - %d / %d\n', iTestFile, nTestFile)

        % load test data
        [tH2Wn,RH2Wn,scanO,~] = loadTestYAML(testFiles{iTestFile});

        for iMethod = 1:nMethod
            fprintf('\t\ttesting method: %s\n', testedMethods{iMethod})
            tO2He = alltO2He(:, iCalibFile, iMethod);
            RO2He = allRO2He(:, :, iCalibFile, iMethod);

            % sphere fitting
            [~,Df] = sphereFit(tO2He, RO2He, tH2Wn, RH2Wn, scanO);
            eSphere(iTestFile, iCalibFile, iMethod) = abs(Df - D);
        end
    end
end

%% Save result
dateTimeSuffix = string(datetime('now'), 'yyMMdd_HHmm');
resultFilename = 'result/real_sphere_' + dateTimeSuffix + '.mat';
save(resultFilename)

%% Print result
fprintf('\n\nSphere reconstruction error:\n')
for iMethod = 1:nMethod
    fprintf('\t%s: ', testedMethods{iMethod})
    for iTestFile = 1:nTestFile
        fprintf('& %.3f ', mean(eSphere(iTestFile, :, iMethod), 'omitnan'))
    end
    fprintf('& %.3f\n', mean(eSphere(:, :, iMethod), 'all', 'omitnan'))
end

%% Paper figure
% select calibration and plane file
iTestFile = 8;
iCalibFile = 7;

% load test data
[tH2Wn,RH2Wn,scanO,~] = loadTestYAML(testFiles{iTestFile});

% number of measurements
m = size(tH2Wn,3);

% number of points to visualize per profile
np = 30;

% create figure
figure('Units','centimeters', 'Position',[2 2 18.2 4], 'Color','w');
% tiled layout
tiledlayout(1, nMethod, 'TileSpacing', 'compact', 'Padding', 'tight')

for iMethod = [5 1 2 3 4]
    nexttile

    % hand-eye transformation
    tO2He = alltO2He(:, iCalibFile, iMethod);
    RO2He = allRO2He(:, :, iCalibFile, iMethod);

    % transform profile points
    scanW = cell(1,m);
    for iScan = 1:m
        scanWi = RH2Wn(:,:,iScan)*(RO2He*scanO{iScan} + tO2He) + tH2Wn(:,:,iScan);  % 3 x mpi
        scanW{iScan} = scanWi(:, round(linspace(1,length(scanWi),np)));
    end
    scanW = [scanW{:}];

    % sphere fitting
    pcW = pointCloud(scanW');
    warning('off')
    sphereModel = pcfitsphere(pcW, 0.1);
    warning('on')
    pCf = sphereModel.Center';
    Rf = sphereModel.Radius;

    % plot sphere
    [X,Y,Z] = sphere(100);
    X = X*Rf + pCf(1);
    Y = Y*Rf + pCf(2);
    Z = Z*Rf + pCf(3);
    surf(X, Y, Z, 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    colormap summer
    axis equal
    hold on

    % plot point cloud
    ax = pcshow(scanW', [0 0.4470 0.7410], 'BackgroundColor', 'w', ...
        'AxesVisibility', 'on', 'MarkerSize', 3);
    view(25,5)
    % hide tick label
    ax.XTickLabel = [];
    ax.YTickLabel = [];
    ax.ZTickLabel = [];

end
