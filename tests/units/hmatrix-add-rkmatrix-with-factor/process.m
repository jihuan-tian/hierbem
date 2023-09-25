clear all;
load hmatrix-add-rkmatrix-with-factor.output;

b = 3.5;
M = M1 + b * M2;
norm(M1 - H_full_before_add, 'fro') / norm(M1, 'fro')
norm(H_full_after_add - M, 'fro') / norm(M, 'fro')
