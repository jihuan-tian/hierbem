clear all;

load "M.dat";
load "xy.dat";
load "hmatrix-Tvmult-tril-serial-iterative.output";

hmat_rel_err = norm(H_full - M, 'fro') / norm(M, 'fro')
y = 0.3 * y0 + 1.5 * transpose(M) * x;
y_rel_err = norm(y1_cpp - y, 2) / norm(y, 2)

hmat_complex_rel_err = norm(H_full_complex - M_complex, 'fro') / norm(M_complex, 'fro')
y1_complex = 0.3 * y0_complex + 1.5 * transpose(M_complex) * x_complex;
y2_complex = complex(0.3, 0.2) * y1_complex + complex(1.5, 2.1) * transpose(M_complex) * x_complex;
y3_complex = complex(0.3, 0.2) * y2_complex + 1.5 * transpose(M_complex) * x_complex;
y4_complex = 0.3 * y3_complex + complex(1.5, 2.1) * transpose(M_complex) * x_complex;

y1_complex_rel_err = norm(y1_cpp_complex - y1_complex, 2) / norm(y1_complex, 2)
y2_complex_rel_err = norm(y2_cpp_complex - y2_complex, 2) / norm(y2_complex, 2)
y3_complex_rel_err = norm(y3_cpp_complex - y3_complex, 2) / norm(y3_complex, 2)
y4_complex_rel_err = norm(y4_cpp_complex - y4_complex, 2) / norm(y4_complex, 2)
