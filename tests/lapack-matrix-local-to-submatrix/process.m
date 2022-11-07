clear all;
load lapack-matrix-local-to-submatrix.output;

tau = (5:12) + 1;
sigma = (7:14) + 1;
M_b_octave = M(tau, sigma);

tau_subset = (7:10) + 1;
sigma_subset = (10:12) + 1;
M_b_submatrix_octave = M(tau_subset, sigma_subset);

norm(M_b_submatrix - M_b_submatrix_octave, 'fro') / norm(M_b_submatrix, 'fro')
