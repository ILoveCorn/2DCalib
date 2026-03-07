function w = pageLogSO3(R)
%PAGELOGSO3
%   Vectorized logarithm map of SO(3) using page operations
%
%   Input:
%       R: rotation matrices        (3 x 3 x m)
%   Output:
%       w: rotation vectors         (3 x 1 x m)

m = size(R, 3);

% Compute trace for all matrices: trace(R) = R11 + R22 + R33
tr = R(1,1,:) + R(2,2,:) + R(3,3,:);  % 1 x 1 x m

% Compute angle: theta = arccos((trace(R) - 1) / 2)
% Clamp to [-1, 1] to avoid numerical issues
cos_theta = (tr - 1) / 2;
cos_theta = min(max(cos_theta, -1), 1);
theta = acos(cos_theta);  % 1 x 1 x m

% Extract skew-symmetric part: (R - R') / 2
% The rotation vector is: w = theta / (2*sin(theta)) * [R32-R23; R13-R31; R21-R12]
w = zeros(3, 1, m);
w(1,1,:) = R(3,2,:) - R(2,3,:);
w(2,1,:) = R(1,3,:) - R(3,1,:);
w(3,1,:) = R(2,1,:) - R(1,2,:);

% Handle small angles and general case
small_angle = abs(theta) < 1e-10;
theta_safe = theta;
theta_safe(small_angle) = 1;  % avoid division by zero

% Coefficient: theta / (2*sin(theta))
% For small angles: use Taylor expansion ~ 0.5 + theta^2/12
coeff = theta ./ (2 * sin(theta_safe));
coeff(small_angle) = 0.5;

% Apply coefficient
w = coeff .* w;
end
