## Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

clear all;

load "M.dat";
load "x.dat";
load "hmatrix-vmult-hermite-symm.output";

MM_complex = tril2fullhsym(M_complex);

factor = 0.5;
factor_complex = complex(0.5, 0.3);

## Here we compare the lower triangular and diagonal part of the matrix instead
## of the complete Hermite symmetric matrix, since the operation in @p tril2fullhsym for
## restoring the complete Hermite symmetric matrix will increase the rounding-off error
## a little bit, which may cause the \hmatrix error limit 1e-14 is exceeded.
hmat_complex_rel_err = norm(H_full_complex - M_complex, 'fro') / norm(M_complex, 'fro')
y_complex = MM_complex * x_complex;
y1_complex_rel_err = norm(y1_complex - y_complex, 2) / norm(y_complex, 2)
y2_complex_rel_err = norm(y2_complex - factor * y_complex, 2) / norm(factor * y_complex, 2)
y3_complex_rel_err = norm(y3_complex - factor_complex * y_complex, 2) / norm(factor_complex * y_complex, 2)
