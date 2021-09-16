clear all;
load lapack-matrix-global-to-submatrix.output;

tau = [2, 5, 7, 10];
sigma = [3, 8, 9, 16];
M_b_octave = M(tau + 1, sigma + 1);
norm(M_b_octave - M_b, 'fro') / norm(M_b, 'fro')
