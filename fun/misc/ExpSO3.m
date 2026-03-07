function SO3 = ExpSO3(so3)
%EXPSO3
%   Exponential map of SO(3)
%
%   Input:
%       so3: rotation vector    (3 x 1)
%   Output:
%       SO3: rotation matrix    (3 x 3)
so3 = so3(:);

% axis-angle
a = so3;
theta = norm(a);

% exponential map
if theta ~= 0
    SO3 = axang2rotm([a', theta]);
else
    SO3 = eye(3);
end

end