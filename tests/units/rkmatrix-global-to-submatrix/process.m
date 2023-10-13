clear all;
load rkmatrix-global-to-submatrix.output;

tau = [2, 3, 4, 5, 7, 10, 18, 19] + 1;
sigma = [3, 4, 8, 9, 11, 13, 15, 16, 17] + 1;
M_b_octave = M(tau, sigma);

norm(M_b_octave - M_b, 'fro') / norm(M_b_octave, 'fro')
