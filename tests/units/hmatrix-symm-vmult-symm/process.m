clear all;

load "M.dat";
load "x.dat";
load "hmatrix-symm-vmult-symm.output";

MM = tril2fullsym(M);

## Here we compare the lower triangular and diagonal part of the matrix instead
## of the complete symmetric matrix, since the operation in @p tril2fullsym for
## restoring the complete symmetric matrix will increase the rounding-off error
## a little bit, which may cause the \hmatrix error limit 1e-14 is exceeded.
hmat_rel_err = norm(H_full - M, 'fro') / norm(M, 'fro')
y = MM * x;
y1_rel_err = norm(y1 - y, 2) / norm(y, 2)
y2_rel_err = norm(y2 - 0.5 * y, 2) / norm(0.5 * y, 2)
