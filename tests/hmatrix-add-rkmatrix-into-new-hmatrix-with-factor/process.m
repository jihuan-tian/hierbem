clear all;
load hmatrix-add-rkmatrix-into-new-hmatrix-with-factor.output;

b = 3.5;
M = M1 + b * M2;
norm(M1 - H_full, 'fro') / norm(M1, 'fro')
norm(H_sum_full - M, 'fro') / norm(M, 'fro')
