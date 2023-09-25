clear all;
load "rkmatrix-frobenius-norm.output";

rkmat = A * B';
norm(rkmat, 'fro');
calculated_value = 7.750767182503684e+04;
(norm(rkmat, 'fro') - calculated_value) / norm(rkmat, 'fro')
