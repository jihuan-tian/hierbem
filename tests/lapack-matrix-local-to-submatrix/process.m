clear all;
load lapack-matrix-local-to-submatrix.output;

tau = [2, 3, 4, 5, 7, 10, 18, 19] + 1;
sigma = [3, 4, 8, 9, 11, 13, 15, 16, 17] + 1;
M_b_octave = M(tau, sigma);

tau_subset = [3, 7, 10, 19] + 1;
sigma_subset = [8, 13, 17] + 1;
M_b_submatrix_octave = M(tau_subset, sigma_subset);

norm(M_b_submatrix - M_b_submatrix_octave, 'fro') / norm(M_b_submatrix, 'fro')
