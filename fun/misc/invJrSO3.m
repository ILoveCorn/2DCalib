function iJr = invJrSO3(so3)
%INVJRSO3
%   Inverse of the right Jacobian of SO(3)
%
%   Input:
%       so3: rotation vector            (3 x 1)
%   Output:
%       iJr: inverse right Jacobian matrix  (3 x 3)
so3 = so3(:);

% axis-angle
a = so3;
theta = norm(a);

% right Jacobian
if abs(theta) > 1e-15
    so3H = hat(so3);
    iJr = eye(3) + 0.5*so3H + (1/theta^2 - (1+cos(theta))/(2*theta*sin(theta)))*so3H*so3H;
else
    iJr = eye(3);
end

end