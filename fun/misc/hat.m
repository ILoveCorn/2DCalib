function vhat = hat(v)
%HAT skew-symmetric matrix of R(3)

vhat = [0 -v(3) v(2)
    v(3) 0 -v(1)
    -v(2) v(1) 0];
end