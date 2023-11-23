clear -x enable_figure;

load "L.dat";
load "b.dat";
load "hmatrix-forward-substitution-coarse-ntp.output";

hmat_rel_err = norm(H_full - L, 'fro') / norm(L, 'fro')

x_octave = forward_substitution(L, b);
x_rel_err = norm(x - x_octave, 'fro') / norm(x_octave, 'fro')
