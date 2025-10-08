function [xc,zc,r] = circleFit(px,pz)
%CIRCLEFIT implementation of https://doi.org/10.1016/j.measurement.2023.112936
%   v1.0 2025-03-19

% reshape input
px = px(:);
pz = pz(:);

%% DLS estimation
A = [px pz ones(size(px))];
b = -px.^2 - pz.^2;
x = A\b;

a = x(1);
b = x(2);
c = x(3);

xc = -a/2;
zc = -b/2;
r = sqrt(a^2+b^2-4*c)/2;

end