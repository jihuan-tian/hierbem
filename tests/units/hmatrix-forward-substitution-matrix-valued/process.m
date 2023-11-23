clear -x enable_figure;

load "L.dat";
load "Z.dat";
load "hmatrix-forward-substitution-matrix-valued.output";

L_rel_err = norm(L_full - L, 'fro') / norm(L, 'fro')
Z_rel_err = norm(Z_full - Z, 'fro') / norm(Z, 'fro')

X_octave = L \ Z;
X_rel_err = norm(X - X_octave, 'fro') / norm(X_octave, 'fro')
