clear all;

load "M.dat";
load "xy.dat";
load "hmatrix-vmult-tril-serial-iterative.output";

hmat_rel_err = norm(H_full - M, 'fro') / norm(M, 'fro')
y1 = 0.3 * y0 + 1.5 * M * x;
y2 = 3.7 * y1 + 8.2 * M * x;
y1_rel_err = norm(y1_cpp - y1, 2) / norm(y1, 2)
y2_rel_err = norm(y2_cpp - y2, 2) / norm(y2, 2)
