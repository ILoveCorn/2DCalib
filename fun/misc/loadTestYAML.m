function [tH2Wn,RH2Wn,scanO,centerSignY] = loadTestYAML(filename)
%LOADTESTYAML Load test data from yaml
%   filename - xxx.yaml
%
%   tH2Wn - hand position in world frame with noise     (3 x 1 x m)
%   RH2Wn - hand orientation in world frame with noise  (3 x 3 x m)
%   scanO - raw scan points                             ({3 x mpi} x m cell)
%   centerSignY - sign of sphere center's y-coordinate  (1 x m)

% load yaml
yamlData = yaml.loadFile(filename);
captures = yamlData.captures;

% number of samples
m = length(captures);

% allocate memory
tH2Wn = zeros(3, 1, m);
qH2Wn = zeros(m, 4);
scanO = cell(1, m);
centerSignY = zeros(1, m);

for i = 1:m
    % hand position
    tH2Wn(:,:,i) = 1000 * [captures{i}.pose.position.x, captures{i}.pose.position.y, captures{i}.pose.position.z]';
    % hand orientation
    qH2Wn(i,:) = [captures{i}.pose.orientation.w, captures{i}.pose.orientation.x, ...
        captures{i}.pose.orientation.y, captures{i}.pose.orientation.z];

    % scan points
    pOx = cell2mat(captures{i}.points.x)';  % 1 x mpi
    pOz = cell2mat(captures{i}.points.z)';
    scanO{i} = [pOx; zeros(size(pOx)); pOz];

    % sphere center side
    if isfield(captures{i}.pose, 'center_side')
        centerSignY(i) = -captures{i}.pose.center_side.sign_y;
    end

RH2Wn = quat2rotm(qH2Wn);
end

