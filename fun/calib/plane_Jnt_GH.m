function [tO2He,RO2He,ne,de,tH2We,RH2We,sigmae,time] = plane_Jnt_GH(tO2He,RO2He,ne,de,sigmae,tH2Wn,RH2Wn,pOn,vOn,iImax,iOmax,tol)
%PLANE_JNT_GH
%   Proposed calibration algorithm using joint optimization and Gauss-Helmert model
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

% max iterations
% iImax = 100;
% iOmax = 20;

% problem dimension
m = size(tH2Wn,3);
nx = 9; % [O2He(6) ne(2) de(1)]
nl = 6; % [tH2Wi(3) wH2Wi(3)]
ng = 2; % [g(2)]

% recover hand pose & vectorize
pOnH = zeros(3,3,m);
vOnH = zeros(3,3,m);
parfor i = 1:m
    pOnH(:,:,i) = hat(pOn(:,i));
    vOnH(:,:,i) = hat(vOn(:,i));
end

% initial hand-world transformation
tH2We = tH2Wn;
RH2We = RH2Wn;

% residue
vt = zeros(3,1,m);             % tH2W
vR = repmat(eye(3),[1 1 m]);   % RH2W

% Jacobians
A = zeros(ng,nx,m);     % dg / dx
B = zeros(ng,nl,m);     % dg / dl
pHeH = zeros(3,3,m);    % hat(pHe)
vHeH = zeros(3,3,m);    % hat(vHe)

% covariance
vare = repelem(sigmae,3).^2;
Cl = diag(vare);
Wl = diag(1./vare);

% increments
dx = zeros(nx,1);
dvar = ones(size(vare));

% redundancy number
rti = zeros(1,m);
rwi = zeros(1,m);

while iO <= iOmax
    %%% estimate unknowns %%%
    while iI == 1 || (norm(dx) > tol && iI <= iImax) || converged
        tic % start clock
        % reprojection
        pHe = pagemtimes(RO2He,pOn) + tO2He;
        pWe = pagemtimes(RH2We,pHe) + tH2We;
        vHe = pagemtimes(RO2He,vOn);
        vWe = pagemtimes(RH2We,vHe);

        % initialize plane normal
        if iI == 1
            % normalize
            ne = ne/norm(ne);
            if ne(1) == 0 && ne(2) == 0
                Rnex = [1 0 0]';
            else
                Rnex = [-ne(2) ne(1) 0]';
                Rnex = Rnex/norm(Rnex);
            end
            Rne = [Rnex hat(ne)*Rnex ne];
        end
        neT = ne';

        % constraints
        G = [pagemtimes(neT,pWe)-de; pagemtimes(neT,vWe)];  % 2 x 1 x m

        % measurement residue
        axangvw = rotm2axang(vR)';
        % TODO: check rotation axis norm
        vw = reshape(axangvw(1:3,:).*axangvw(4,:),3,1,m);
        V = [vt; vw];   % 6 x 1 x m

        % Jacobians
        % dg1 / dx
        A(1,1:3,:) = pagemtimes(neT,RH2We);
        A(1,4:6,:) = -pagemtimes(neT,pagemtimes(RH2We,pagemtimes(RO2He,pOnH)));
        Jne = -Rne*hat([0 0 1]);
        A(1,7,:) = pagemtimes(Jne(:,1)',pWe);
        A(1,8,:) = pagemtimes(Jne(:,2)',pWe);
        A(1,9,:) = -1;
        % dg2 / dx
        A(2,4:6,:) = -pagemtimes(neT,pagemtimes(RH2We,pagemtimes(RO2He,vOnH)));
        A(2,7,:) = pagemtimes(Jne(:,1)',vWe);
        A(2,8,:) = pagemtimes(Jne(:,2)',vWe);   % 2 x nx x m

        % dg1 / dl
        B(1,1:3,:) = neT(:,:,ones(1,m));
        parfor i = 1:m
            pHeH(:,:,i) = hat(pHe(:,:,i));
            vHeH(:,:,i) = hat(vHe(:,:,i));

            % right Jacobian of dR
            JrvR(:,:,i) = JrSO3(vw(:,:,i));
        end
        B(1,4:6,:) = -pagemtimes(neT,pagemtimes(RH2We,pHeH));
        % dg2 / dl
        B(2,4:6,:) = -pagemtimes(neT,pagemtimes(RH2We,vHeH));   % 2 x nl x m

        % (43a), D = B
        B(:,4:6,:) = pagemtimes(B(:,4:6,:),JrvR);

        % Hessian & gradient
        % (43b)
        E = pagemtimes(B,pagemtimes(Cl,'none',B,'transpose')); % 2 x 2 x m
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
        % unknowns
        tO2He = tO2He + dx(1:3);
        RO2He = RO2He*ExpSO3(dx(4:6));
        Rne = Rne*ExpSO3([dx(7:8); 0]);
        ne = Rne(:,3);
        de = de + dx(9);
        % measurements
        vt = vt + dl(1:3,:,:);
        tH2We = tH2Wn + vt;
        parfor i = 1:m
            vR(:,:,i) = vR(:,:,i)*ExpSO3(dl(4:6,:,i));
        end
        RH2We = pagemtimes(RH2Wn,vR);

        % add time
        time = time + toc;
        iI = iI + 1;

        % termination criteria
        converged = ~(norm(dx(1:6)) > tol && iI <= iImax);
    end

    %%% estimate covariance %%%
    tic % start clock
    % diagonal of redundancy matrix
    DtE_D = pagemtimes(DtE_,B);
    DtE_A = pagemtimes(DtE_,A);
    Rdiag = pagemtimes(Cl,DtE_D-pagemtimes(pagemrdivide(DtE_A,H),'none',DtE_A,'transpose'));
    Rt = Rdiag(1:3,1:3,:);
    Rw = Rdiag(4:6,4:6,:);
    % redundancy number
    parfor i = 1:m
        rti(i) = trace(Rt(:,:,i));
        rwi(i) = trace(Rw(:,:,i));
    end
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

