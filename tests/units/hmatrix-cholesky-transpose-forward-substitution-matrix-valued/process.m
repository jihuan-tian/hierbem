clear all;

load_packages;
load "L.dat";
load "Z.dat";
load "hmatrix-cholesky-transpose-forward-substitution-matrix-valued.output";

X_octave = (L \ Z')';
x_rel_err = norm(X - X_octave, 'fro') / norm(X_octave, 'fro')
