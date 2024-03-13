clear all;

load "M.dat";
load "x.dat";
load "hmatrix-tvmult-tril.output";

hmat_rel_err = norm(H_full - M, 'fro') / norm(M, 'fro')
y = M' * x;
y1_rel_err = norm(y1 - y, 2) / norm(y, 2)
y2_rel_err = norm(y2 - 0.5 * y, 2) / norm(0.5 * y, 2)
