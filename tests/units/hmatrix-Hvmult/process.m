clear all;

load "M.dat";
load "x.dat";
load "hmatrix-Hvmult.output";

factor = 0.5;
factor_complex = complex(0.5, 0.3);

hmat_complex_rel_err = norm(H_full_complex - M_complex, 'fro') / norm(M_complex, 'fro')
y_complex = ctranspose(M_complex) * x_complex;
y1_complex_rel_err = norm(y1_complex - y_complex, 2) / norm(y_complex, 2)
y2_complex_rel_err = norm(y2_complex - factor * y_complex, 2) / norm(factor * y_complex, 2)
y3_complex_rel_err = norm(y3_complex - factor_complex * y_complex, 2) / norm(factor_complex * y_complex, 2)
