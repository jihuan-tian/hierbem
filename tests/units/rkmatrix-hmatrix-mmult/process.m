clear all;
load rkmatrix-hmatrix-mmult.output;

norm(M1 - M1_rk.A * M1_rk.B', 'fro') / norm(M1, 'fro')

M = M1 * M2;
norm(M - M_rk.A * M_rk.B', 'fro') / norm(M, 'fro')
