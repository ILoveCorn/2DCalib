function [tO2He,RO2He,ne,time] = plane_IROS2015(tO2He,RO2He,tH2Wn,RH2Wn,pOn,tol)
%PLANE_IROS2015 implementation of https://doi.org/10.1109/IROS.2015.7353884
%   Scheme: PCA -> two-step
%   Parameterization: <SO(3),R(3)>
%   Solver: linear least square
%   v1.0 2025-03-14
%
%   tH2Wn - hand position in world frame with noise     (3x1xmxmP)
%   RH2Wn - hand orientation in world frame with noise  (3x3xmxmP)
%   pOn   - a point on intersection line in OCT frame   (3x1xmxmP)

% control variables
i = 1;
time = 0;

% number of measurements
m = size(tH2Wn,3);
% number of planes
mP = size(tH2Wn,4);

while true
    tic % start clock
    %%% reprojection %%%
    pWe = pagemtimes(RH2Wn,pagemtimes(RO2He,pOn) + tO2He) + tH2Wn;  % 3*1*m*mP

    %%% plane normal %%%
    pWavg = reshape(mean(pWe,3),3,mP);
    neT = zeros(1,3,m,mP);
    de = zeros(1,mP);
    for iP = 1:mP
        pWei = reshape(pWe(:,:,:,iP),3,m);
        [~,~,V] = svd((pWei-pWavg(:,iP))',0);
        neT(:,:,:,iP) = repmat(V(:,3)',[1 1 m]);
        de(:,iP) = V(:,3)'*pWavg(:,iP);
    end

    %%% rotation %%%
    Ap = pagemtimes(neT,[pagemtimes(RH2Wn,pOn(1,:,:,:)) pagemtimes(RH2Wn,pOn(3,:,:,:)) RH2Wn]); % 1*9*m*mP
    A = reshape(permute(Ap,[1 3 4 2]),m*mP,9);
    qp = pagemtimes(neT,tH2Wn); % 1*1*m*mP
    q = reshape(qp,m*mP,1);
    Y = reshape(repmat(de,[m 1]),m*mP,1) - q;

    % solve
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

ne = pagetranspose(neT(:,:,1,:));

end

