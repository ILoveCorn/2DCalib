function [tH2Wn,RH2Wn,pOn,vOn,scanO] = loadPlaneYAML(filename)
%loadPlaneYAML
%   Load plane calibration data from yaml for OCT / LiDAR (obsolete)
%
%   Input:
%       filename: xxx.yaml
%
%   Output:
%       tH2Wn: noisy hand-world position            (3 x 1 x m)
%       RH2Wn: noisy hand-world orientation         (3 x 3 x m)
%       pOn: intersection position in OCT frame     (3 x 1 x m)
%       vOn: intersection direction in OCT frame    (3 x 1 x m)
%       scanO: target profile points                (3 x mp x m)

% load yaml
yamlData = yaml.loadFile(filename);
captures = yamlData.captures;

% number of measurements
m = length(captures);

% allocate memory
tH2Wn = zeros(3,1,m);
qH2Wn = zeros(m,4);
pOn = zeros(3,1,m);
vOn = zeros(3,1,m);

% check data type: lidar / oct
if startsWith(filename, 'lidar', 'IgnoreCase', true)
    dataType = 'lidar';
elseif startsWith(filename, 'oct', 'IgnoreCase', true)
    dataType = 'oct';
else
    error('Unknown data type: filename must start with \"lidar\" or \"oct\".');
end

% sensor-specific settings
if dataType == "lidar"
    % LiDAR parameters
    thetaFull = 135;
    npFull = 1081;

    % number of points per scan
    mp = length(captures{1}.averaged_scan);
    thetaMax = (mp-1) / (npFull-1) * thetaFull;
    theta = linspace(-thetaMax, thetaMax, mp);  % 1 x mp
    scanO = zeros(3, mp, m);
else
    mpOCT = 5000;
    scanOcell = cell(m, 1);
end

% data processing
for i = 1:m
    % hand position
    tH2Wn(:,:,i) = 1000 * [captures{i}.pose.position.x, captures{i}.pose.position.y, captures{i}.pose.position.z]'; % original measurements in meters

    % hand orientation
    qH2Wn(i,:) = [captures{i}.pose.orientation.w, captures{i}.pose.orientation.x, captures{i}.pose.orientation.y, captures{i}.pose.orientation.z];

    if dataType == "lidar"
        % convert range to point coordinates
        range = 1000 * cell2mat(captures{i}.averaged_scan)';  % 1 x mp
        pOx = -range .* sind(theta);
        pOz = range .* cosd(theta);

        scanO(1,:,i) = pOx;
        scanO(3,:,i) = pOz;
    else
        pOx = cell2mat(captures{i}.points.x)';  % 1 x mpi
        pOz = cell2mat(captures{i}.points.z)';
        scanOcell{i} = [pOx; zeros(size(pOx)); pOz];

        if length(pOx) < mpOCT
            mpOCT = length(pOx);
        end
    end

    % intersection position
    pOn(:,:,i) = [mean(pOx) 0 mean(pOz)]';

    % intersection direction
    pInt = polyfit(pOx,pOz,1);
    intTheta = atan(pInt(1));
    vOn(:,:,i) = [cos(intTheta) 0 sin(intTheta)]';
end

% unify scan length of OCT
if dataType == "oct"
    scanO = zeros(3,mpOCT,m);
    for i = 1:length(scanOcell)
        mpi = size(scanOcell{i}, 2);
        if mpi > mpOCT
            % uniformly extract mpOCT columns
            scanO(:, :, i) = scanOcell{i}(:, round(linspace(1, mpi, mpOCT)));
        else
            scanO(:, :, i) = scanOcell{i};
        end
    end
end

RH2Wn = quat2rotm(qH2Wn);
end

