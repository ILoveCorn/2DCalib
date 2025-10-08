function SO3 = ExpSO3(so3)
%ExpSO3 exponential map of SO(3)
%   so(3): [rx, ry, rz]
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