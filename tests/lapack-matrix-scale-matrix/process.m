clear all;
load lapack-matrix-scale-matrix.output;

b = 3.5;
norm(A_after_scaling - A_before_scaling * b, 'fro') / norm(A_before_scaling * b, 'fro')
