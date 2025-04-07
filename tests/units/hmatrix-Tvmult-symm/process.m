clear all;

load "M.dat";
load "x.dat";
load "hmatrix-Tvmult-symm.output";

MM = tril2fullsym(M);
MM_complex = tril2fullsym(M_complex);

factor = 0.5;
factor_complex = complex(0.5, 0.3);

## Here we compare the lower triangular and diagonal part of the matrix instead
## of the complete symmetric matrix, since the operation in @p tril2fullsym for
## restoring the complete symmetric matrix will increase the rounding-off error
## a little bit, which may cause the \hmatrix error limit 1e-14 is exceeded.
hmat_rel_err = norm(H_full - M, 'fro') / norm(M, 'fro')
y = MM * x;
y1_rel_err = norm(y1 - y, 2) / norm(y, 2)
y2_rel_err = norm(y2 - factor * y, 2) / norm(factor * y, 2)

hmat_complex_rel_err = norm(H_full_complex - M_complex, 'fro') / norm(M_complex, 'fro')
y_complex = MM_complex * x_complex;
y1_complex_rel_err = norm(y1_complex - y_complex, 2) / norm(y_complex, 2)
y2_complex_rel_err = norm(y2_complex - factor * y_complex, 2) / norm(factor * y_complex, 2)
y3_complex_rel_err = norm(y3_complex - factor_complex * y_complex, 2) / norm(factor_complex * y_complex, 2)
