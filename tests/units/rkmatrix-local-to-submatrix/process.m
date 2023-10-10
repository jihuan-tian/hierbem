clear all;
load rkmatrix-local-to-submatrix.output;

tau_subset = [3, 7, 10, 19] + 1;
sigma_subset = [8, 13, 17] + 1;
M_b_submatrix_octave = M(tau_subset, sigma_subset);

norm(M_b_submatrix_octave - M_b_submatrix, 'fro') / norm(M_b_submatrix_octave)
