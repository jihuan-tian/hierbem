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
load "xy.dat";
load "hmatrix-Hvmult-triu-serial-iterative.output";

hmat_complex_rel_err = norm(H_full_complex - M_complex, 'fro') / norm(M_complex, 'fro')
y1_complex = 0.3 * y0_complex + 1.5 * ctranspose(M_complex) * x_complex;
y2_complex = complex(0.3, 0.2) * y1_complex + complex(1.5, 2.1) * ctranspose(M_complex) * x_complex;
y3_complex = complex(0.3, 0.2) * y2_complex + 1.5 * ctranspose(M_complex) * x_complex;
y4_complex = 0.3 * y3_complex + complex(1.5, 2.1) * ctranspose(M_complex) * x_complex;

y1_complex_rel_err = norm(y1_cpp_complex - y1_complex, 2) / norm(y1_complex, 2)
y2_complex_rel_err = norm(y2_cpp_complex - y2_complex, 2) / norm(y2_complex, 2)
y3_complex_rel_err = norm(y3_cpp_complex - y3_complex, 2) / norm(y3_complex, 2)
y4_complex_rel_err = norm(y4_cpp_complex - y4_complex, 2) / norm(y4_complex, 2)
