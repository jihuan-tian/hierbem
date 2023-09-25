clear all;
load lapack-matrix-add-with-factor.output;

b = 3.5;
norm(C - (A + b * B), 'fro') / norm(A + b * B, 'fro')
norm(A_self_added - (A + b * B), 'fro') / norm(A + b * B, 'fro')
