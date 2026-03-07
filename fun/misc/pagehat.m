function S = pagehat(v)
%PAGEHAT
%   Vectorized skew-symmetric matrix using page operations
%
%   Input:
%       v: vectors              (3 x 1 x m)
%   Output:
%       S: skew-symmetric matrices  (3 x 3 x m)

m = size(v, 3);
S = zeros(3, 3, m);
S(1,2,:) = -v(3,1,:);
S(1,3,:) = v(2,1,:);
S(2,1,:) = v(3,1,:);
S(2,3,:) = -v(1,1,:);
S(3,1,:) = -v(2,1,:);
S(3,2,:) = v(1,1,:);
end
