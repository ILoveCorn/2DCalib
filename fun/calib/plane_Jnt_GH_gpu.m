function [tO2He,RO2He,ne,de,tH2We,RH2We,sigmae,time] = plane_Jnt_GH_gpu(tO2He,RO2He,ne,de,sigmae,tH2Wn,RH2Wn,pOn,vOn,iImax,iOmax,tol)
%PLANE_JNT_GH_GPU
%   Proposed calibration algorithm using joint optimization and Gauss-Helmert model
%   GPU-accelerated version using gpuArray
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
nx = 9; % [O2He(6) ne(2) de(1)]
nl = 6; % [tH2Wi(3) wH2Wi(3)]
ng = 2; % [g(2)]

% recover hand pose & vectorize (on CPU first, then transfer to GPU)
pOnH = zeros(3,3,m);
vOnH = zeros(3,3,m);
for i = 1:m
    pOnH(:,:,i) = hat(pOn(:,i));
    vOnH(:,:,i) = hat(vOn(:,i));
end

% Transfer data to GPU
tO2He = gpuArray(tO2He);
RO2He = gpuArray(RO2He);
ne = gpuArray(ne);
de = gpuArray(de);
tH2Wn = gpuArray(tH2Wn);
RH2Wn = gpuArray(RH2Wn);
pOn = gpuArray(pOn);
vOn = gpuArray(vOn);
pOnH = gpuArray(pOnH);
vOnH = gpuArray(vOnH);

% initial hand-world transformation
tH2We = tH2Wn;
RH2We = RH2Wn;

% residue
vt = gpuArray(zeros(3,1,m));             % tH2W
vR = gpuArray(repmat(eye(3),[1 1 m]));   % RH2W

% Jacobians
A = gpuArray(zeros(ng,nx,m));     % dg / dx
B = gpuArray(zeros(ng,nl,m));     % dg / dl

% Precompute constant GPU arrays to avoid repeated allocation
I3_gpu = gpuArray(eye(3));
Z001_hat = gpuArray([0 -1 0; 1 0 0; 0 0 0]);  % hat([0 0 1])

% covariance
vare = gpuArray(repelem(sigmae,3).^2);
Cl = gpuArray(diag(gather(vare)));
Wl = gpuArray(diag(1./gather(vare)));

% increments
dx = gpuArray(zeros(nx,1));
dvar = gpuArray(ones(size(vare)));

% redundancy number
% rti = gpuArray(zeros(1,m));
% rwi = gpuArray(zeros(1,m));

while iO <= iOmax
    %%% estimate unknowns %%%
    while iI == 1 || (norm(gather(dx)) > tol && iI <= iImax) || converged
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
            ne_cpu = gather(ne);
            if ne_cpu(1) == 0 && ne_cpu(2) == 0
                Rnex = gpuArray([1 0 0]');
            else
                Rnex = gpuArray([-ne_cpu(2) ne_cpu(1) 0]');
                Rnex = Rnex/norm(Rnex);
            end
            % inline hat
            %
            %  avoid function call overhead
            ne_hat = [0 -ne(3) ne(2); ne(3) 0 -ne(1); -ne(2) ne(1) 0];
            Rne = [Rnex ne_hat*Rnex ne];
        end
        neT = ne';

        % constraints
        G = [pagemtimes(neT,pWe)-de; pagemtimes(neT,vWe)];  % 2 x 1 x m

        % measurement residue (GPU-accelerated LogSO3)
        vw = pageLogSO3_gpu(vR);  % 3 x 1 x m
        V = [vt; vw];   % 6 x 1 x m

        % Jacobians
        % dg1 / dx
        A(1,1:3,:) = pagemtimes(neT,RH2We);
        A(1,4:6,:) = -pagemtimes(neT,pagemtimes(RH2We,pagemtimes(RO2He,pOnH)));
        Jne = -Rne*Z001_hat;
        A(1,7,:) = pagemtimes(Jne(:,1)',pWe);
        A(1,8,:) = pagemtimes(Jne(:,2)',pWe);
        A(1,9,:) = -1;
        % dg2 / dx
        A(2,4:6,:) = -pagemtimes(neT,pagemtimes(RH2We,pagemtimes(RO2He,vOnH)));
        A(2,7,:) = pagemtimes(Jne(:,1)',vWe);
        A(2,8,:) = pagemtimes(Jne(:,2)',vWe);   % 2 x nx x m

        % dg1 / dl - vectorized hat computation
        pHeH = pagehat_gpu(pHe);  % 3x3xm
        vHeH = pagehat_gpu(vHe);  % 3x3xm

        % right Jacobian of dR - vectorized
        JrvR = pageJrSO3_gpu(vw, I3_gpu);  % 3x3xm

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
        % unknowns
        tO2He = tO2He + dx(1:3);
        RO2He = RO2He*ExpSO3_gpu(dx(4:6), I3_gpu);
        Rne = Rne*ExpSO3_gpu([dx(7:8); 0], I3_gpu);
        ne = Rne(:,3);
        de = de + dx(9);
        % measurements
        vt = vt + dl(1:3,:,:);
        tH2We = tH2Wn + vt;
        % vectorized ExpSO3 for all measurements
        dl_R = dl(4:6,:,:);
        ExpR = pageExpSO3_gpu(dl_R, I3_gpu);  % 3x3xm
        vR = pagemtimes(vR, ExpR);
        RH2We = pagemtimes(RH2Wn,vR);

        % add time
        time = time + toc;
        iI = iI + 1;

        % termination criteria
        converged = ~(norm(gather(dx(1:6))) > tol && iI <= iImax);
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
    sigmae = sqrt(gather(vare([1 4])));
    Cl = diag(vare);
    Wl = diag(1./vare);

    % add time
    time = time + toc;

    % check convergence
    if max(abs(gather(dvar)-1)) < 1e-4 || iO == iOmax
        break
    else
        % reset control variables
        vt = gpuArray(zeros(3,1,m));
        vR = gpuArray(repmat(eye(3),[1 1 m]));
        tH2We = tH2Wn;
        RH2We = RH2Wn;
        iO = iO + 1;
        iI = 1;
        converged = false;
    end

end

% Transfer results back to CPU
tO2He = gather(tO2He);
RO2He = gather(RO2He);
ne = gather(ne);
de = gather(de);
tH2We = gather(tH2We);
RH2We = gather(RH2We);

end

%% GPU-compatible helper functions (optimized vectorized versions)

function S = pagehat_gpu(v)
%PAGEHAT_GPU Vectorized skew-symmetric matrix for 3x1xm input
%   Input: v (3 x 1 x m)
%   Output: S (3 x 3 x m)
m = size(v, 3);
S = zeros(3, 3, m, 'like', v);
S(1,2,:) = -v(3,1,:);
S(1,3,:) = v(2,1,:);
S(2,1,:) = v(3,1,:);
S(2,3,:) = -v(1,1,:);
S(3,1,:) = -v(2,1,:);
S(3,2,:) = v(1,1,:);
end

function R = ExpSO3_gpu(w, I3)
%EXPSO3_GPU Exponential map SO(3) for single vector (GPU version)
%   Reuses preallocated identity matrix I3
theta = norm(w);
if theta < 1e-10
    R = I3;
else
    w1 = w(1); w2 = w(2); w3 = w(3);
    w_hat = [0 -w3 w2; w3 0 -w1; -w2 w1 0];
    R = I3 + sin(theta)/theta * w_hat + (1-cos(theta))/(theta^2) * (w_hat * w_hat);
end
end

function R = pageExpSO3_gpu(w, I3)
%PAGEEXPSO3_GPU Vectorized exponential map SO(3) for 3x1xm input
%   Input: w (3 x 1 x m), I3 (3 x 3) identity matrix on GPU
%   Output: R (3 x 3 x m)
m = size(w, 3);

% Compute theta for all
theta = sqrt(sum(w.^2, 1));  % 1 x 1 x m
theta = reshape(theta, 1, 1, m);

% Compute hat matrices
w_hat = pagehat_gpu(w);  % 3 x 3 x m
w_hat2 = pagemtimes(w_hat, w_hat);  % 3 x 3 x m

% Coefficients with small angle handling
small_angle = theta < 1e-10;
theta_safe = theta;
theta_safe(small_angle) = 1;  % avoid division by zero

c1 = sin(theta_safe) ./ theta_safe;
c2 = (1 - cos(theta_safe)) ./ (theta_safe.^2);
c1(small_angle) = 1;
c2(small_angle) = 0.5;

% Broadcast coefficients
R = repmat(I3, 1, 1, m) + c1 .* w_hat + c2 .* w_hat2;
end

function Jr = pageJrSO3_gpu(w, I3)
%PAGEJRSO3_GPU Vectorized right Jacobian of SO(3) for 3x1xm input
%   Input: w (3 x 1 x m), I3 (3 x 3) identity matrix on GPU
%   Output: Jr (3 x 3 x m)
m = size(w, 3);

% Compute theta for all
theta = sqrt(sum(w.^2, 1));  % 1 x 1 x m
theta = reshape(theta, 1, 1, m);

% Compute hat matrices
w_hat = pagehat_gpu(w);  % 3 x 3 x m
w_hat2 = pagemtimes(w_hat, w_hat);  % 3 x 3 x m

% Coefficients with small angle handling
small_angle = theta < 1e-10;
theta_safe = theta;
theta_safe(small_angle) = 1;  % avoid division by zero

c1 = (1 - cos(theta_safe)) ./ (theta_safe.^2);
c2 = (theta_safe - sin(theta_safe)) ./ (theta_safe.^3);
c1(small_angle) = 0.5;
c2(small_angle) = 1/6;

% Broadcast coefficients
Jr = repmat(I3, 1, 1, m) - c1 .* w_hat + c2 .* w_hat2;
end

function w = pageLogSO3_gpu(R)
%PAGELOGSO3_GPU Vectorized logarithm map of SO(3) for 3x3xm input (GPU version)
%   Input: R (3 x 3 x m) - rotation matrices on GPU
%   Output: w (3 x 1 x m) - rotation vectors (axis * angle)
%
%   The logarithm map extracts the rotation vector from a rotation matrix:
%   theta = arccos((trace(R) - 1) / 2)
%   w = (theta / (2*sin(theta))) * [R32-R23; R13-R31; R21-R12]

m = size(R, 3);

% Compute trace for all matrices: trace(R) = R11 + R22 + R33
tr = R(1,1,:) + R(2,2,:) + R(3,3,:);  % 1 x 1 x m

% Compute angle: theta = arccos((trace(R) - 1) / 2)
% Clamp to [-1, 1] to avoid numerical issues
cos_theta = (tr - 1) / 2;
cos_theta = min(max(cos_theta, -1), 1);
theta = acos(cos_theta);  % 1 x 1 x m

% Extract skew-symmetric part: (R - R') / 2
% The rotation vector is: w = theta / (2*sin(theta)) * [R32-R23; R13-R31; R21-R12]
w = zeros(3, 1, m, 'like', R);
w(1,1,:) = R(3,2,:) - R(2,3,:);
w(2,1,:) = R(1,3,:) - R(3,1,:);
w(3,1,:) = R(2,1,:) - R(1,2,:);

% Handle small angles and general case
small_angle = abs(theta) < 1e-10;
theta_safe = theta;
theta_safe(small_angle) = 1;  % avoid division by zero

% Coefficient: theta / (2*sin(theta))
% For small angles: use Taylor expansion ~ 0.5 + theta^2/12
coeff = theta ./ (2 * sin(theta_safe));
coeff(small_angle) = 0.5;

% Apply coefficient
w = coeff .* w;
end
