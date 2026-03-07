function w = randSO3()
%RANDSO3
%   Uniform sampling from SO(3)
%
%   Output:
%       w: rotation vector  (3 x 1)

% uniformly sample w from open sphere with radius of pi
theta = 2*pi*rand;
phi = asin(2*rand-1);
radii = pi*(rand^(1/3));
[wx,wy,wz] = sph2cart(theta,phi,radii);
w = [wx wy wz]';

while rand >= (2-2*cos(norm(w)))/(norm(w)^2)
    % repeat sampling
    theta = 2*pi*rand;
    phi = asin(2*rand-1);
    radii = pi*(rand^(1/3));
    [wx,wy,wz] = sph2cart(theta,phi,radii);
    w = [wx wy wz]';
end

end