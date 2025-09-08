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
load "hmatrix-Tvmult-triu.output";

factor = 0.5;
factor_complex = complex(0.5, 0.3);

hmat_rel_err = norm(H_full - M, 'fro') / norm(M, 'fro')
y = transpose(M) * x;
y1_rel_err = norm(y1 - y, 2) / norm(y, 2)
y2_rel_err = norm(y2 - factor * y, 2) / norm(factor * y, 2)

hmat_complex_rel_err = norm(H_full_complex - M_complex, 'fro') / norm(M_complex, 'fro')
y_complex = transpose(M_complex) * x_complex;
y1_complex_rel_err = norm(y1_complex - y_complex, 2) / norm(y_complex, 2)
y2_complex_rel_err = norm(y2_complex - factor * y_complex, 2) / norm(factor * y_complex, 2)
y3_complex_rel_err = norm(y3_complex - factor_complex * y_complex, 2) / norm(factor_complex * y_complex, 2)
