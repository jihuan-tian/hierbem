clear all;
load rkmatrix-global-to-rkmatrix.output;

tau = [2, 3, 4, 5, 7, 10, 18, 19] + 1;
sigma = [3, 4, 8, 9, 11, 13, 15, 16, 17] + 1;
M_b_octave = M(tau, sigma);

norm(M - M_rk.A * M_rk.B', 'fro') / norm(M, 'fro')
norm(M_b_octave - M_b_rk.A * M_b_rk.B', 'fro') / norm(M_b_octave, 'fro')
norm(M_b_octave - M_b_rk1.A * M_b_rk1.B', 'fro') / norm(M_b_octave, 'fro')
