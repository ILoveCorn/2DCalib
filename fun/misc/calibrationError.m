function [et,eR] = calibrationError(t,R,te,Re)
%CALIBRATIONERROR
%   Calculate calibration error
%
%   Input:
%       t: ground-truth hand-eye position       (3 x 1)
%       R: ground-truth hand-eye orientation    (3 x 3)
%       te: estimated hand-eye position         (3 x 1)
%       Re: estimated hand-eye orientation      (3 x 3)
%   Output:
%       et: position error                      [mm]
%       eR: orientation error                   [degree]

et = norm(t-te);

axangeR = rotm2axang(Re'*R);
eR = rad2deg(axangeR(4));
end

