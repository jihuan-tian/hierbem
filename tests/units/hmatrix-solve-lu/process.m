% Clear all vars except enable_figure
clear -x enable_figure;

load "hmatrix-solve-lu.output";
load "M.dat";
load "b.dat";

% Calculate relative error between H-Matrix and full matrix based on Frobenius-norm
hmat_rel_err = norm(H_full - M, 'fro') / norm(M, 'fro')

% Calculate relative error between H-matrix and full matrix solution based on 2-norm
x_octave = M \ b;
x_rel_err = norm(x_octave - x, 2) / norm(x_octave, 2)
