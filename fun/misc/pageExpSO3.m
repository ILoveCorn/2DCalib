function R = pageExpSO3(w)
%PAGEEXPSO3
%   Vectorized exponential map of SO(3) using page operations
%
%   Input:
%       w: rotation vectors     (3 x 1 x m)
%   Output:
%       R: rotation matrices    (3 x 3 x m)

m = size(w, 3);

% Compute theta for all
theta = sqrt(sum(w.^2, 1));  % 1 x 1 x m
theta = reshape(theta, 1, 1, m);

% Compute hat matrices
w_hat = zeros(3, 3, m);
w_hat(1,2,:) = -w(3,1,:);
w_hat(1,3,:) = w(2,1,:);
w_hat(2,1,:) = w(3,1,:);
w_hat(2,3,:) = -w(1,1,:);
w_hat(3,1,:) = -w(2,1,:);
w_hat(3,2,:) = w(1,1,:);

w_hat2 = pagemtimes(w_hat, w_hat);  % 3 x 3 x m

% Coefficients with small angle handling
small_angle = theta < 1e-10;
theta_safe = theta;
theta_safe(small_angle) = 1;  % avoid division by zero

c1 = sin(theta_safe) ./ theta_safe;
c2 = (1 - cos(theta_safe)) ./ (theta_safe.^2);
c1(small_angle) = 1;
c2(small_angle) = 0.5;

% Broadcast coefficients
I3 = repmat(eye(3), 1, 1, m);
R = I3 + c1 .* w_hat + c2 .* w_hat2;
end
