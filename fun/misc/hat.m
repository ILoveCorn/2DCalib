function vhat = hat(v)
%HAT
%   Skew-symmetric matrix of R(3)
%
%   Input:
%       v: 3D vector                    (3 x 1)
%   Output:
%       vhat: skew-symmetric matrix     (3 x 3)

vhat = [0 -v(3) v(2)
    v(3) 0 -v(1)
    -v(2) v(1) 0];
end