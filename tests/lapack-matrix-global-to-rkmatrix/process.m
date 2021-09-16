clear all;
format long;
load lapack-matrix-global-to-rkmatrix.output;

tau = [2, 5, 7, 10] + 1;
sigma = [3, 8, 9, 16] + 1;
M_b = M(tau, sigma);

norm(M_b - rkmat_no_trunc.A * rkmat_no_trunc.B', 'fro') / norm(M_b, 'fro')
norm(M_b - rk1mat.A * rk1mat.B', 'fro') / norm(M_b, 'fro')
