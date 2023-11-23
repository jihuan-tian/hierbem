clear -x enable_figure;

load "U.dat";
load "b.dat";
load "hmatrix-backward-substitution-coarse-ntp.output";

hmat_rel_err = norm(H_full - U, 'fro') / norm(U, 'fro')

x_octave = backward_substitution(U, b);
x_rel_err = norm(x - x_octave, 'fro') / norm(x_octave, 'fro')
