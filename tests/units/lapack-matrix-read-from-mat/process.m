clear all;
load lapack-matrix-read-from-mat.output;
load input.mat;

norm(M - M_read, 'fro') / norm(M, 'fro')
