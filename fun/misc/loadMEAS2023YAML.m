function [tH2Wn,RH2Wn,pcOe,scanO] = loadMEAS2023YAML(filename,mt,D)
%loadMEAS2023YAML load calibration data of MEAS2023
%
%   mR - number of orientations
%   mt - number of samples per orientation
%
%   tH2Wn - hand position in world frame with noise     (3 x mt x mR)
%   RH2Wn - hand orientation in world frame with noise  (3 x 3 x mR)
%   pcOe - estimated sphere center in OCT frame (3 x mt x m)
%   scanO - raw scan points ({3 x npi} x mt x m)
%
%   D - diameter of calibration sphere

[tH2Wn,RH2Wn,scanO,centerSignY] = loadTestYAML(filename);

m = size(tH2Wn, 3); % m = mt x mR
mR = m / mt;

% DEBUG
% RH2Wn1 = RH2Wn(:, :, 1:mt:end);
% RH2Wn2 = RH2Wn(:, :, 2:mt:end);
% RH2Wn3 = RH2Wn(:, :, 3:mt:end);
%
% dR12 = pagemtimes(RH2Wn1, 'none', RH2Wn2, 'transpose');
% dR13 = pagemtimes(RH2Wn1, 'none', RH2Wn3, 'transpose');
%
% axang12 = rotm2axang(dR12);
% axang13 = rotm2axang(dR13);
%
% ang12 = rad2deg(axang12(:, 4));
% ang13 = rad2deg(axang13(:, 4));

% radius of calibration sphere
R = D/2;

% reshape data
tH2Wn = reshape(tH2Wn, 3, mt, mR);
RH2Wn = RH2Wn(:, :, 1:mt:end);
scanO = reshape(scanO, mt, mR);
centerSignY = reshape(centerSignY, mt, mR);

% estimated sphere center
pcOe = zeros(3, mt, mR);

% sphere fitting
for iR = 1:mR
    for it = 1:mt
        [pcOex,pcOez,rOe] = circleFit(scanO{it, iR}(1, :), scanO{it, iR}(3, :));
        % handle fitting error
        if rOe > R
            rOe = R;
        end
        pcOey = centerSignY(it, iR) * sqrt(R^2 - rOe^2);
        pcOe(:, it, iR) = [pcOex, pcOey, pcOez]';
    end
end

end

