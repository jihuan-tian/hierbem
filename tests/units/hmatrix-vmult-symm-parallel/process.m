clear all;

load "M.dat";
load "xy.dat";
load "hmatrix-vmult-symm-parallel.output";

MM = tril2fullsym(M);

## Here we compare the lower triangular and diagonal part of the matrix instead
## of the complete symmetric matrix, since the operation in @p tril2fullsym for
## restoring the complete symmetric matrix will increase the rounding-off error
## a little bit, which may cause the \hmatrix error limit 1e-14 is exceeded.
hmat_rel_err = norm(H_full - M, 'fro') / norm(M, 'fro')
y1 = 0.3 * y0 + 1.5 * MM * x;
y2 = 3.7 * y1 + 8.2 * MM * x;
y1_rel_err = norm(y1_cpp - y1, 2) / norm(y1, 2)
y2_rel_err = norm(y2_cpp - y2, 2) / norm(y2, 2)
