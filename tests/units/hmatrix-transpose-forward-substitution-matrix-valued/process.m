clear -x enable_figure;
load "U.dat";
load "Z.dat";
load "hmatrix-transpose-forward-substitution-matrix-valued.output";

U_rel_err = norm(U_full - U, 'fro') / norm(U, 'fro')
Z_rel_err = norm(Z_full - Z, 'fro') / norm(Z, 'fro')

X_octave = (U' \ Z')';
X_rel_err = norm(X - X_octave, 'fro') / norm(X_octave, 'fro')
