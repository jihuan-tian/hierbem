clear all;
load hmatrix-hmatrix-mmult-all-fine-ntp-with-adding.output;

M = M1 * M2 + M1;
norm(M - H_full, 'fro') / norm(M, 'fro')