clear all;

load "M.dat";
load "xy.dat";
load "hmatrix-Hvmult-symm-serial-iterative.output";

MM_complex = tril2fullsym(M_complex);

## Here we compare the lower triangular and diagonal part of the matrix instead
## of the complete symmetric matrix, since the operation in @p tril2fullsym for
## restoring the complete symmetric matrix will increase the rounding-off error
## a little bit, which may cause the \hmatrix error limit 1e-14 is exceeded.
hmat_complex_rel_err = norm(H_full_complex - M_complex, 'fro') / norm(M_complex, 'fro')
y1_complex = 0.3 * y0_complex + 1.5 * ctranspose(MM_complex) * x_complex;
y2_complex = complex(0.3, 0.2) * y1_complex + complex(1.5, 2.1) * ctranspose(MM_complex) * x_complex;
y3_complex = complex(0.3, 0.2) * y2_complex + 1.5 * ctranspose(MM_complex) * x_complex;
y4_complex = 0.3 * y3_complex + complex(1.5, 2.1) * ctranspose(MM_complex) * x_complex;

y1_complex_rel_err = norm(y1_cpp_complex - y1_complex, 2) / norm(y1_complex, 2)
y2_complex_rel_err = norm(y2_cpp_complex - y2_complex, 2) / norm(y2_complex, 2)
y3_complex_rel_err = norm(y3_cpp_complex - y3_complex, 2) / norm(y3_complex, 2)
y4_complex_rel_err = norm(y4_cpp_complex - y4_complex, 2) / norm(y4_complex, 2)
