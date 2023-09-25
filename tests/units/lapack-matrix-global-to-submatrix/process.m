clear all;
load lapack-matrix-global-to-submatrix.output;

tau = [7,8,9,10] + 1;
sigma = [3,4,5,6] + 1;
M_b_octave = M(tau, sigma);
norm(M_b_octave - M_b, 'fro') / norm(M_b, 'fro')
