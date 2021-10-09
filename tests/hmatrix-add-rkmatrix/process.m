clear all;
load hmatrix-add-rkmatrix.output;

M = M1 + M2;
norm(M1 - H_full_before_add, 'fro') / norm(M1, 'fro')
norm(H_full_after_add - M, 'fro') / norm(M, 'fro')
