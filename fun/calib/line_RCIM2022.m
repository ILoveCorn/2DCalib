function [tO2He,RO2He,time] = line_RCIM2022(tH2Wn,RH2Wn,pOn)
%LINE_RCIM2022 implementation of https://doi.org/10.1016/j.rcim.2021.102221
%   Scheme: PCA -> two-step
%   Parameterization: <SO(3),R(3)>
%   Solver: linear least square
%   v1.0 2025-02-10
%
%   mt    - number of measurements per orientation
%   tH2Wn - hand position in world frame with noise     (3xmtxm)
%   RH2Wn - hand orientation in world frame with noise  (3x3xm)
%   pOn   - intersection in OCT frame with noise        (3xmtxm)

% number of measurements
mt = size(pOn,2);
m = size(pOn,3);

tic % start clock
%%% orientation %%%
Ar = zeros(3*m*(mt-2),9);
br = zeros(3*m*(mt-2),1);
for i = 1:m
    % build us and ut
    us = diff(pOn(:,:,i),1,2);
    ut = diff(tH2Wn(:,:,i),1,2);

    % build Ar and br
    for j = 1:mt-2
        usi = us(:,j);  usj = us(:,j+1);
        uti = ut(:,j);  utj = ut(:,j+1);
        Rb = RH2Wn(:,:,i);

        A1 = kron(cross(usi,usj)',Rb);
        A2 = hat(utj)'*kron(usi',Rb);
        A3 = hat(uti)'*kron(usj',Rb);
        Ar((1:3)+3*(mt-2)*(i-1)+3*(j-1),:) = A1+A2-A3;
        br((1:3)+3*(mt-2)*(i-1)+3*(j-1),:) = -cross(uti,utj);
    end
end

% solve for rs
Rs = reshape(Ar\br,3,3);

% orthogonalization
[U2,~,V2] = svd(Rs);
D = diag([1 1 det(U2)*det(V2)]);
Rsa = U2*D*V2';


%%% position %%%
% estimate line direction
ub = zeros(3,m*(mt-1));
for i = 1:m
    % build us and ut
    us = diff(pOn(:,:,i),1,2);
    ut = diff(tH2Wn(:,:,i),1,2);
    % build ub
    ub(:,(1:(mt-1))+(mt-1)) = RH2Wn(:,:,i)*Rsa*us + ut;
end
% SVD for ub*
[~,~,V] = svd(ub',0);
uba = V(:,1);
ubaHat = hat(uba);

% build At and bt
At = zeros(3*mt*(m-1),3);
bt = zeros(3*mt*(m-1),1);
for i = 1:mt
    % bt
    for j = 1:m-1
        tbj = tH2Wn(:,i,j+1);
        tbi = tH2Wn(:,i,j);
        Rbj = RH2Wn(:,:,j+1);
        Rbi = RH2Wn(:,:,j);
        psj = pOn(:,i,j+1);
        psi = pOn(:,i,j);

        if i == 1   % only build At once
            At((1:3)+3*(m-1)*(i-1)+3*(j-1),:) = ubaHat'*(RH2Wn(:,:,j+1)-RH2Wn(:,:,j));
        end

        bt((1:3)+3*(m-1)*(i-1)+3*(j-1),:) = ubaHat*(Rbj*Rsa*psj-Rbi*Rsa*psi+tbj-tbi);
    end

    if i > 1
        At(3*(m-1)*(i-1)+1:3*(m-1)*i,:) = At(1:3*(m-1),:);
    end
end

% solve for Ts
Ts = At\bt;
time = toc;

tO2He = Ts;
RO2He = Rsa;

end
