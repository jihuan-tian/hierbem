clear all;
load fullmatrix-hmatrix-mmult.output;

M = M1 * M2;
norm(M - M_full, 'fro') / norm(M, 'fro')
norm(M - M_rk.A * M_rk.B', 'fro') / norm(M, 'fro')
norm(M_full - M_rk.A * M_rk.B', 'fro') / norm(M_full, 'fro')
