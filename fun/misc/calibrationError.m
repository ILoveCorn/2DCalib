function [et,eR] = calibrationError(t,R,te,Re)
%CALIBRATIONERROR calculate calibration error
%   v1.0 2025-03-21
%   Unit: [mm, degree]

et = norm(t-te);

axangeR = rotm2axang(Re'*R);
eR = rad2deg(axangeR(4));
end

