% measure.m
%   calculate hand-eye transformation from measured marker position

clearvars
close all
clc

% calibrated (marker frame)
p0Mcalib = [0, 0, 0]';
p1Mcalib = [16.0948, 25.5177, 0]';
p2Mcalib = [29.8033, -18.3393, 0.7324]';
p3Mcalib = [64.9506, 0, 0]';

% scanned (OCT frame)
p0Oscan = [3.1104, -42.6799, -87.0274]';
p1Oscan = [28.5399, -41.6264, -70.9775]';
p2Oscan = [-15.2175, -41.0304, -57.3173]';
p3Oscan = [3.5092, -38.9583, -22.2190]';

% marker's orientation in OCT frame
% x-axis
rx = p3Oscan - p0Oscan;
rx = rx / norm(rx);
% y-axis
v01 = p1Oscan - p0Oscan;
ry = v01 - v01'*rx*rx;
ry = ry / norm(ry);
% hand-eye orientation
RO2H = [rx ry cross(rx,ry)]';

% hand-eye position
tO2H = -RO2H*p0Oscan;

% scanned (marker frame)
p0Mscan = RO2H*p0Oscan + tO2H;
p1Mscan = RO2H*p1Oscan + tO2H;
p2Mscan = RO2H*p2Oscan + tO2H;
p3Mscan = RO2H*p3Oscan + tO2H;

%% Save result
dateTimeSuffix = string(datetime('now'), 'yyMMdd_HHmm');
resultFilename = "result/measure_" + dateTimeSuffix + ".mat";
save(resultFilename, "tO2H","RO2H")