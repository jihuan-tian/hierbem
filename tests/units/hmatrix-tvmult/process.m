clear all;
load "hmatrix-tvmult.output";
y_trans_tilde = M' * x;
norm(y_trans_tilde - y_trans1, 2) / norm(y_trans1)
y_trans1 ./ y_trans2
