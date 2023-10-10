clear all;
load hmatrix-add-rkmatrix-into-new-hmatrix.output;

M = M1 + M2;
norm(M1 - H_full, 'fro') / norm(M1, 'fro')
norm(H_sum_full - M, 'fro') / norm(M, 'fro')
