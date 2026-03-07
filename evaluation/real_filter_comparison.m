% real_filter_comparison.m
%   Compare pOn and vOn between original and filtered data
%   pOn: intersection line position
%   vOn: intersection line unit direction vector

clearvars
close all
clc

%% Load data
originalFile = 'oct_checkerboard_250714_2030_distorted.yaml';
filteredFile = 'oct_checkerboard_250714_2030.yaml';

fprintf('=== Filter Comparison: pOn and vOn ===\n\n');

% load data
[~,~,pOnOrig,vOnOrig,~] = loadPlaneYAML(originalFile);
[~,~,pOnFilt,vOnFilt,~] = loadPlaneYAML(filteredFile);

% number of measurements
m = size(pOnOrig, 3);
fprintf('Number of measurements: %d\n\n', m);

%% Compute differences
pOnDiff2D = zeros(2, m);    % position difference in 2D [x; z] (mm)
vOnAngleError = zeros(m, 1); % vOn angle error with sign (deg)

for i = 1:m
    % extract 2D position difference (x, z) since y=0
    pOnDiff2D(:, i) = [pOnOrig(1,:,i) - pOnFilt(1,:,i); ...
        pOnOrig(3,:,i) - pOnFilt(3,:,i)];

    % angle error with sign (using cross product to determine sign)
    dotProd = dot(vOnOrig(:,:,i), vOnFilt(:,:,i));
    % clamp to [-1, 1] to handle numerical errors
    dotProd = max(-1, min(1, dotProd));
    crossProd = cross(vOnOrig(:,:,i), vOnFilt(:,:,i));
    angleRad = acos(max(-1, min(1, dotProd)));
    % use sign of y-component of cross product to determine rotation direction
    vOnAngleError(i) = rad2deg(angleRad) * sign(crossProd(2));
end

%% Statistics
fprintf('--- Position Difference (pOn) ---\n');
fprintf('X direction:\n');
fprintf('  Mean:   %.4f um\n', 1000*mean(pOnDiff2D(1,:)));
fprintf('  Std:    %.4f um\n', 1000*std(pOnDiff2D(1,:)));
fprintf('  Min:    %.4f um\n', 1000*min(pOnDiff2D(1,:)));
fprintf('  Max:    %.4f um\n', 1000*max(pOnDiff2D(1,:)));
fprintf('  Median: %.4f um\n\n', 1000*median(pOnDiff2D(1,:)));

fprintf('Z direction:\n');
fprintf('  Mean:   %.4f um\n', 1000*mean(pOnDiff2D(2,:)));
fprintf('  Std:    %.4f um\n', 1000*std(pOnDiff2D(2,:)));
fprintf('  Min:    %.4f um\n', 1000*min(pOnDiff2D(2,:)));
fprintf('  Max:    %.4f um\n', 1000*max(pOnDiff2D(2,:)));
fprintf('  Median: %.4f um\n\n', 1000*median(pOnDiff2D(2,:)));

fprintf('--- Direction Angle Error (vOn with sign) ---\n');
fprintf('  Mean:   %.4f deg\n', mean(vOnAngleError));
fprintf('  Std:    %.4f deg\n', std(vOnAngleError));
fprintf('  Min:    %.4f deg\n', min(vOnAngleError));
fprintf('  Max:    %.4f deg\n', max(vOnAngleError));
fprintf('  Median: %.4f deg\n\n', median(vOnAngleError));

%% Visualization
% Create figure with 1x2 subplots
figure('Units', 'centimeters', 'Position', [2 2 17.6 8.8]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% 1. pOn z direction difference histogram
nexttile;
histogram(1000*pOnDiff2D(2,:), 'Normalization', 'probability');
xlabel('$z\ [\mathrm{\mu m}]$', 'Interpreter', 'latex', 'FontSize', 11);
ylabel('Frequency', 'Interpreter', 'latex', 'FontSize', 11);
title('Position-$z$ difference distribution', 'Interpreter', 'latex', 'FontSize', 12);
grid on;
% xline(1000*mean(pOnDiff2D(2,:)), 'k--', 'LineWidth', 0.5);

% 2. vOn angle error with sign (histogram)
nexttile;
histogram(vOnAngleError, 'Normalization', 'probability');
xlabel('Angle\ [$^\circ$]', 'Interpreter', 'latex', 'FontSize', 11);
ylabel('Frequency', 'Interpreter', 'latex', 'FontSize', 11);
title('Direction difference distribution', 'Interpreter', 'latex', 'FontSize', 12);
grid on;
% xline(mean(vOnAngleError), 'k--', 'LineWidth', 0.5);

%% Distribution type test
% Test whether the differences follow a normal distribution using
% Jarque-Bera test, one-sample Kolmogorov-Smirnov test, and
% Anderson-Darling test. Significance level alpha = 0.05.
% H0: data comes from a normal distribution.

alpha = 0.05;
diffLabels = {'pOn-z (um)', 'vOn angle (deg)'};
diffData = {1000*pOnDiff2D(2,:), vOnAngleError'};

fprintf('=== Distribution Type Tests (alpha = %.2f) ===\n\n', alpha);

for k = 1:numel(diffData)
    x = diffData{k};
    xStd = (x - mean(x)) / std(x);  % standardize for KS test

    fprintf('--- %s ---\n', diffLabels{k});

    % 1. Jarque-Bera test (tests skewness and kurtosis)
    [hJB, pJB] = jbtest(x, alpha);
    fprintf('  Jarque-Bera:       p = %.4f, H0 %s\n', pJB, ...
        ternary(hJB, 'REJECTED (non-normal)', 'not rejected (normal)'));

    % 2. One-sample Kolmogorov-Smirnov test against standard normal
    [hKS, pKS] = kstest(xStd, 'Alpha', alpha);
    fprintf('  Kolmogorov-Smirnov: p = %.4f, H0 %s\n', pKS, ...
        ternary(hKS, 'REJECTED (non-normal)', 'not rejected (normal)'));

    % 3. Anderson-Darling test
    [hAD, pAD] = adtest(x, 'Alpha', alpha);
    fprintf('  Anderson-Darling:  p = %.4f, H0 %s\n', pAD, ...
        ternary(hAD, 'REJECTED (non-normal)', 'not rejected (normal)'));

    % 4. Lilliefors test (modified KS test, does not require known params)
    [hLF, pLF] = lillietest(x, 'Alpha', alpha);
    fprintf('  Lilliefors:        p = %.4f, H0 %s\n', pLF, ...
        ternary(hLF, 'REJECTED (non-normal)', 'not rejected (normal)'));

    % Summary: majority vote
    nReject = hJB + hKS + hAD + hLF;
    if nReject >= 3
        fprintf('  >> Conclusion: likely NON-NORMAL distribution\n\n');
    elseif nReject <= 1
        fprintf('  >> Conclusion: likely NORMAL distribution\n\n');
    else
        fprintf('  >> Conclusion: INCONCLUSIVE (2/4 tests reject)\n\n');
    end
end

% Q-Q plots for visual normality assessment
figure('Units', 'centimeters', 'Position', [2 12 17.6 8.8]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

qqLabels = {'$\Delta p_z\ [\mathrm{\mu m}]$', ...
    '$\Delta \theta\ [^\circ]$'};
qqTitles = {'Q-Q: Position-$z$ diff.', ...
    'Q-Q: Direction diff.'};

for k = 1:2
    nexttile;
    qqplot(diffData{k});
    xlabel('Standard normal quantiles', 'Interpreter', 'latex', 'FontSize', 10);
    ylabel(qqLabels{k}, 'Interpreter', 'latex', 'FontSize', 10);
    title(qqTitles{k}, 'Interpreter', 'latex', 'FontSize', 11);
    grid on;
end

%% Save result
% dateTimeSuffix = string(datetime('now'), 'yyMMdd_HHmm');
% resultFilename = 'result/real_filter_comparison_' + dateTimeSuffix + '.mat';
% save(resultFilename, 'pOnDiff2D', 'vOnAngleError');
% fprintf('Results saved to: %s\n', resultFilename);

%% Helper function
function s = ternary(cond, sTrue, sFalse)
% Ternary operator helper for concise conditional string selection
if cond
    s = sTrue;
else
    s = sFalse;
end
end

