clear all;
load "hmatrix-vmult.output";
y_tilde = M * x;
norm(y_tilde - y1, 2) / norm(y1)
y1 ./ y2
