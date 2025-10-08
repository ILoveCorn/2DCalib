% real_plane.m
%   plane reconstruction experiments

clearvars
close all
clc

%% Load calibration result
load('result/real_calibration_251008_1022.mat', 'testedMethods', 'planeFiles', 'alltO2He', 'allRO2He', 'Time')

% get test files
testFiles = planeFiles;

nMethod = length(testedMethods);
nCalibFile = length(testFiles);
nTestFile = nCalibFile;

%% Plane reconstruction
ePlane = zeros(nTestFile, nCalibFile, nMethod);

for iCalibFile = 1:nCalibFile
    fprintf('Plane reconstruction using calibration - %d / %d\n', iCalibFile, nCalibFile)

    for iTestFile = 1:nTestFile
        fprintf('\tusing plane data - %d / %d\n', iTestFile, nTestFile)

        % load test data
        [tH2Wn,RH2Wn,~,~,scanO] = loadPlaneYAML(testFiles{iTestFile});

        for iMethod = 1:nMethod
            fprintf('\t\ttesting method: %s\n', testedMethods{iMethod})
            tO2He = alltO2He(:, iCalibFile, iMethod);
            RO2He = allRO2He(:, :, iCalibFile, iMethod);
            switch iMethod
                case {1,2,3} % plane-based methods
                    if iTestFile == iCalibFile  % skip data used for calibration
                        ePlane(iTestFile, iCalibFile, iMethod) = NaN;
                    else
                        [~,~,ePlane(iTestFile, iCalibFile, iMethod)] = planeFit(tO2He,RO2He,tH2Wn,RH2Wn,scanO);
                    end
                case {4,5}  % non plane-based methods
                    [~,~,ePlane(iTestFile, iCalibFile, iMethod)] = planeFit(tO2He,RO2He,tH2Wn,RH2Wn,scanO);
            end
        end
    end
end

%% Save result
dateTimeSuffix = string(datetime('now'), 'yyMMdd_HHmm');
resultFilename = 'result/real_plane_' + dateTimeSuffix + '.mat';
save(resultFilename)

%% Print result
fprintf('\n\nPlane reconstruction error:\n')
for iMethod = 1:nMethod
    fprintf('\t%s: ', testedMethods{iMethod})
    for iTestFile = 1:nTestFile
        fprintf('& %.3f ', mean(ePlane(iTestFile, :, iMethod), 'omitnan'))
    end
    fprintf('& %.3f\n', mean(ePlane(:, :, iMethod), 'all', 'omitnan'))
end
