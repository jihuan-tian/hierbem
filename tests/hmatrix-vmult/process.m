clear all;
load hmatrix-vmult.output;
y_tilde = M * x;
norm(y_tilde - y, 2) / norm(y)
y_trans_tilde = M' * x;
norm(y_trans_tilde - y_trans, 2) / norm(y_trans)
