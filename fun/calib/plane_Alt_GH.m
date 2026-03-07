function [tO2He,RO2He,ne,de,tH2We,RH2We,sigmae,time] = plane_Alt_GH(tO2He,RO2He,ne,de,sigmae,tH2Wn,RH2Wn,pOn,vOn,iImax,iOmax,tol)
%PLANE_ALT_GH
%   Alternating calibration algorithm using Gauss-Helmert model
%   Inner loop optimizes hand-eye parameters only; plane parameters are
%   determined from the estimated hand-eye parameters via SVD
%
%   Input:
%       tO2He: initial hand-eye position      (3 x 1)
%       RO2He: initial hand-eye orientation   (3 x 3)
%       ne: initial plane target normal       (3 x 1)
%       de: initial plane target distance
%       sigmae: initial variance factors [sigma_t sigma_R]
%       tH2Wn: noisy hand-world position            (3 x 1 x m)
%       RH2Wn: noisy hand-world orientation         (3 x 3 x m)
%       pOn: intersection position in OCT frame     (3 x 1 x m)
%       vOn: intersection direction in OCT frame    (3 x 1 x m)
%       iImax, iOmax: max iterations of inner, outer loop
%       tol: convergence tolerance
%   Output:
%       tO2He: estimated hand-eye position      (3 x 1)
%       RO2He: estimated hand-eye orientation   (3 x 3)
%       ne: estimated plane target normal       (3 x 1)
%       de: estimated plane target distance
%       tH2We: corrected hand-world position    (3 x 1 x m)
%       RH2We: corrected hand-world orientation (3 x 3 x m)
%       sigmae: estimated variance factors [sigma_t sigma_R]
%       time: computation time

% control variables
iI = 1; % inner loop
iO = 1; % outer loop
time = 0;
converged = false;  % is converged (for calculate residue & Jacobians of the final estimation)

% problem dimension
m = size(tH2Wn,3);
nx = 6; % [O2He(6)] - only hand-eye parameters
nl = 6; % [tH2Wi(3) wH2Wi(3)]
ng = 2; % [g(2)]

% recover hand pose & vectorize
pOnH = pagehat(pOn);  % 3x3xm
vOnH = pagehat(vOn);  % 3x3xm

% initial hand-world transformation
tH2We = tH2Wn;
RH2We = RH2Wn;

% residue
vt = zeros(3,1,m);             % tH2W
vR = repmat(eye(3),[1 1 m]);   % RH2W

% Jacobians
A = zeros(ng,nx,m);     % dg / dx (only hand-eye parameters)
B = zeros(ng,nl,m);     % dg / dl

% covariance
vare = repelem(sigmae,3).^2;
Cl = diag(vare);
Wl = diag(1./vare);

% increments
dx = zeros(nx,1);
dvar = ones(size(vare));

while iO <= iOmax
    %%% estimate unknowns %%%
    while iI == 1 || (norm(dx) > tol && iI <= iImax) || converged
        tic % start clock
        % reprojection
        pHe = pagemtimes(RO2He,pOn) + tO2He;
        pWe = pagemtimes(RH2We,pHe) + tH2We;
        vHe = pagemtimes(RO2He,vOn);
        vWe = pagemtimes(RH2We,vHe);

        % estimate plane parameters from current hand-eye estimate via SVD
        % reshape to 3 x m matrices
        pWe_mat = reshape(pWe, 3, m);  % 3 x m
        vWe_mat = reshape(vWe, 3, m);  % 3 x m

        % combine directions and de-centered positions for SVD
        pWe_center = mean(pWe_mat, 2);  % 3 x 1
        pWe_centered = pWe_mat - pWe_center;  % 3 x m

        % stack vWe and centered pWe for SVD
        data_mat = [vWe_mat, pWe_centered];  % 3 x 2m

        % SVD to find plane normal (smallest left singular vector)
        % For 3 x 2m matrix: U is 3x3, S is 3x3, V is 2m x 3
        % ne should be in the null space of data_mat', i.e., U(:,end)
        [U, ~, ~] = svd(data_mat, 'econ');
        ne = U(:, end);  % normal is the last left singular vector (3 x 1)

        % ensure consistent normal direction
        if ne' * [0; 0; 1] < 0
            ne = -ne;
        end

        % compute plane distance from mean position
        de = ne' * pWe_center;

        neT = ne';

        % constraints
        G = [pagemtimes(neT,pWe)-de; pagemtimes(neT,vWe)];  % 2 x 1 x m

        % measurement residue - vectorized LogSO3
        vw = pageLogSO3(vR);  % 3 x 1 x m
        V = [vt; vw];   % 6 x 1 x m

        % Jacobians (only for hand-eye parameters)
        % dg1 / dx
        A(1,1:3,:) = pagemtimes(neT,RH2We);
        A(1,4:6,:) = -pagemtimes(neT,pagemtimes(RH2We,pagemtimes(RO2He,pOnH)));
        % dg2 / dx
        A(2,4:6,:) = -pagemtimes(neT,pagemtimes(RH2We,pagemtimes(RO2He,vOnH)));
        % Note: no Jacobian for ne, de since they are not optimized

        % dg1 / dl - vectorized hat computation
        pHeH = pagehat(pHe);  % 3x3xm
        vHeH = pagehat(vHe);  % 3x3xm

        % right Jacobian of dR - vectorized
        JrvR = pageJrSO3(vw);  % 3x3xm

        B(1,1:3,:) = neT(:,:,ones(1,m));
        B(1,4:6,:) = -pagemtimes(neT,pagemtimes(RH2We,pHeH));
        % dg2 / dl
        B(2,4:6,:) = -pagemtimes(neT,pagemtimes(RH2We,vHeH));   % 2 x nl x m

        % (43a), D = B
        B(:,4:6,:) = pagemtimes(B(:,4:6,:),JrvR);

        % Hessian & gradient
        % (43b)
        E = pagemtimes(B,pagemtimes(Cl,'none',B,'transpose'));  % 2 x 2 x m
        % (43c)
        cg = pagemtimes(B,V) - G;   % 2 x 1 x m
        % (43d)
        AtE_ = pagemrdivide(pagetranspose(A),E);    % nx x 2 x m
        Hi = pagemtimes(AtE_,A);    % nx x nx x m
        gi = pagemtimes(AtE_,cg);   % nx x 1 x m
        H = sum(Hi,3);  % nx*nx
        g = sum(gi,3);  % nx*1

        % exit inner loop
        if converged
            break
        end

        % increments
        % (43d)
        dx = H\g;
        % (43e)
        DtE_ = pagemrdivide(pagectranspose(B),E);
        dl = pagemtimes(Cl,pagemtimes(DtE_,cg-pagemtimes(A,dx))) - V;
        dl(4:6,:,:) = pagemtimes(JrvR,dl(4:6,:,:));

        % retraction
        % unknowns (only hand-eye parameters)
        tO2He = tO2He + dx(1:3);
        RO2He = RO2He*ExpSO3(dx(4:6));
        % Note: ne and de are updated via SVD at the beginning of each iteration

        % measurements
        vt = vt + dl(1:3,:,:);
        tH2We = tH2Wn + vt;
        % vectorized ExpSO3 for all measurements
        ExpR = pageExpSO3(dl(4:6,:,:));  % 3x3xm
        vR = pagemtimes(vR, ExpR);
        RH2We = pagemtimes(RH2Wn,vR);

        % add time
        time = time + toc;
        iI = iI + 1;

        % termination criteria
        converged = ~(norm(dx) > tol && iI <= iImax);
    end

    %%% estimate covariance %%%
    tic % start clock
    % diagonal of redundancy matrix
    DtE_D = pagemtimes(DtE_,B);
    DtE_A = pagemtimes(DtE_,A);
    Rdiag = pagemtimes(Cl,DtE_D-pagemtimes(pagemrdivide(DtE_A,H),'none',DtE_A,'transpose'));
    Rt = Rdiag(1:3,1:3,:);
    Rw = Rdiag(4:6,4:6,:);
    % redundancy number - vectorized trace computation
    rti = Rt(1,1,:) + Rt(2,2,:) + Rt(3,3,:);
    rwi = Rw(1,1,:) + Rw(2,2,:) + Rw(3,3,:);
    rt = sum(rti);
    rw = sum(rwi);

    % position residue
    dvar(1:3) = sum(pagemtimes(vt,'transpose',pagemtimes(Wl(1:3,1:3),vt),'none')) / rt;
    % orientation residue
    dvar(4:6) = sum(pagemtimes(vw,'transpose',pagemtimes(Wl(4:6,4:6),vw),'none')) / rw;

    % update covariance
    vare = dvar.*vare;
    sigmae = sqrt(vare([1 4]));
    Cl = diag(vare);
    Wl = diag(1./vare);

    % add time
    time = time + toc;

    % check convergence
    if max(abs(dvar-1)) < 1e-4 || iO == iOmax
        break
    else
        % reset control variables
        vt = zeros(3,1,m);
        vR = repmat(eye(3),[1 1 m]);
        tH2We = tH2Wn;
        RH2We = RH2Wn;
        iO = iO + 1;
        iI = 1;
        converged = false;
    end

end

end
