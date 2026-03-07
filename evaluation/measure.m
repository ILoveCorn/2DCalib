%%% Measured fiducial position and hand-eye transformation

% calibrated (marker frame)
f0M = [0, 0, 0]';
f1M = [16.0948, 25.5177, 0]';
f2M = [29.8033, -18.3393, 0.7324]';
f3M = [64.9506, 0, 0]';

% scanned fiducial positions in OCT frame (4 fiducials x 3 coords x nScan)
f0O = [
    3.1104,   -42.6799, -87.0274;  % scan 0
    1.29919,  -42.4782, -87.461;   % scan 1
    2.5494,   -42.3743, -87.0523;  % scan 2
    2.51745,  -42.3462, -87.4179;  % scan 3
    2.52488,  -42.4878, -87.4371;  % scan 4
    2.52406,  -42.4206, -87.6144;  % scan 5
    2.52386,  -42.3666, -87.4691;  % scan 6
    2.52889,  -42.4933, -87.4553;  % scan 7
    2.53256,  -42.5138, -87.5716;  % scan 8
    2.52538,  -42.5957, -87.6873;  % scan 9
    ]';

f1O = [
    28.5399,  -41.6264, -70.9775;
    27.2112,  -41.6051, -72.3843;
    28.1776,  -41.3046, -71.4391;
    28.2033,  -41.3861, -71.9518;
    28.2083,  -41.2814, -71.8358;
    28.2447,  -41.448,  -72.0302;
    28.247,   -41.4094, -71.9537;
    28.2594,  -41.5239, -71.8789;
    28.2603,  -41.4601, -72.0663;
    28.26,    -41.5248, -72.0966;
    ]';

f2O = [
    -15.2175, -41.0304, -57.3173;
    -15.8531, -40.9114, -57.1356;
    -15.1566, -40.8886, -56.9255;
    -15.2044, -40.9527, -57.3949;
    -15.2264, -41.0588, -57.2932;
    -15.2096, -41.0475, -57.545;
    -15.229,  -40.9941, -57.3382;
    -15.2264, -41.082,  -57.2966;
    -15.2266, -41.0011, -57.5319;
    -15.247,  -41.0392, -57.5201;
    ]';

f3O = [
    3.5092,   -38.9583, -22.2190;
    4.1064,   -39.1397, -22.7288;
    4.16286,  -38.8133, -22.355;
    4.17217,  -39.004,  -22.7192;
    4.13009,  -38.9451, -22.6767;
    4.17,     -39.0028, -22.8047;
    4.16339,  -38.9911, -22.6356;
    4.17274,  -39.0041, -22.5697;
    4.18218,  -38.9543, -22.6799;
    4.15872,  -39.0069, -22.8696;
    ]';

nScan = size(f0O, 2);
tO2H = zeros(3, 1, nScan);
RO2H = zeros(3, 3, nScan);

% Point sets for SVD-based registration
fxM = [f0M, f1M, f2M, f3M];  % 3 x 4, fiducial positions in M(H) frame

for i = 1:nScan
    % Fiducial positions in O frame for the i-th scan
    fxO = [f0O(:,i), f1O(:,i), f2O(:,i), f3O(:,i)];  % 3 x 4

    % Centroids
    cM = mean(fxM, 2);  % 3 x 1
    cO = mean(fxO, 2);  % 3 x 1

    % Centered point sets
    fxMc = fxM - cM;
    fxOc = fxO - cO;

    % Cross-covariance matrix H = fxOc * fxMc'
    H = fxOc * fxMc';  % 3 x 3

    % SVD: H = U * S * V'
    [U, ~, V] = svd(H);

    % Rotation (ensure proper rotation with det = +1)
    D = diag([1, 1, det(V * U')]);
    RO2H(:,:,i) = V * D * U';

    % Translation from centroids
    tO2H(:,:,i) = cM - RO2H(:,:,i) * cO;
end

%% Save result
dateTimeSuffix = string(datetime('now'), 'yyMMdd_HHmm');
resultFilename = "result/measure_" + dateTimeSuffix + ".mat";
save(resultFilename, "tO2H","RO2H")