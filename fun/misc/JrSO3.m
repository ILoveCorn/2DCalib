function Jr = JrSO3(so3)
%JRSO3
%   Right Jacobian of SO(3)
%
%   Input:
%       so3: rotation vector        (3 x 1)
%   Output:
%       Jr: right Jacobian matrix   (3 x 3)
so3 = so3(:);

% axis-angle
a = so3;
theta = norm(a);
so3H = hat(so3);

% right Jacobian
if abs(theta) > 1e-15
    Jr = eye(3) - (1-cos(theta))/theta^2*so3H + (theta-sin(theta))/theta^3*so3H*so3H;
else
    Jr = eye(3);
end

end