function [tO2He,RO2He,ne,time] = plane_RCIM2020(tO2He,RO2He,tH2Wn,RH2Wn,pOn,tol)
%PLANE_RCIM2020
%   Implementation of https://doi.org/10.1016/j.rcim.2019.101823
%
%   Input:
%       tO2He: initial hand-eye position            (3 x 1)
%       RO2He: initial hand-eye orientation         (3 x 3)
%       tH2Wn: hand-world position with noise       (3 x 1 x m)
%       RH2Wn: hand-world orientation with noise    (3 x 3 x m)
%       pOn: intersection in OCT frame              (3 x 1 x m)
%       tol: convergence tolerance
%   Output:
%       tO2He: estimated hand-eye position          (3 x 1)
%       RO2He: estimated hand-eye orientation       (3 x 3)
%       ne: estimated plane normal                  (3 x 1)
%       time: computation time

% control variables
i = 1;
time = 0;

% number of measurements
m = size(tH2Wn,3);

while true
    tic % start clock
    %%% reprojection %%%
    pWe = pagemtimes(RH2Wn,pagemtimes(RO2He,pOn) + tO2He) + tH2Wn;

    %%% plane normal %%%
    pWavg = mean(pWe,3);
    % PCA using SVD
    [~,~,V] = svd((reshape(pWe,3,m)-pWavg)',0);
    neT = V(:,3)';
    de = neT*pWavg;

    %%% rotation %%%
    Ap = pagemtimes(neT,[pagemtimes(RH2Wn,pOn(1,:,:)) pagemtimes(RH2Wn,pOn(3,:,:)) RH2Wn]);
    A = reshape(permute(Ap,[1 3 2]),m,9);
    qp = pagemtimes(neT,tH2Wn);
    q = reshape(qp,m,1);
    Y = de - q;
    w = A\Y; % [Rx Rz t]
    RO2HeOld = RO2He;
    RO2He = [w(1:3) -cross(w(1:3),w(4:6)) w(4:6)];  % [Rx Ry Rz]
    
    % orthogonalization
    [U,~,V] = svd(RO2He);
    D = diag([1 1 det(U)*det(V)]);
    RO2He = U*D*V';

    %%% translation %%%
    tO2HeOld = tO2He;
    tO2He = A(:,7:9)\(Y - A(:,1:6)*[RO2He(:,1); RO2He(:,3)]);
    
    % add in time
    time = time + toc;

    % check convergence
    if i > 1
        dt = tO2He - tO2HeOld;
        daxang = rotm2axang(RO2He'*RO2HeOld);
        if hypot(norm(dt),abs(daxang(4))) < tol || i > 500
            break
        end
    end
    
    % add iteration
    i = i + 1;
end

ne = neT';

end

