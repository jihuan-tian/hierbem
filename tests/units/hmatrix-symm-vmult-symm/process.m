## Copyright (C) 2024 Jihuan Tian <jihuan_tian@hotmail.com>
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
load "hmatrix-symm-vmult-symm.output";

MM = tril2fullsym(M);

## Here we compare the lower triangular and diagonal part of the matrix instead
## of the complete symmetric matrix, since the operation in @p tril2fullsym for
## restoring the complete symmetric matrix will increase the rounding-off error
## a little bit, which may cause the \hmatrix error limit 1e-14 is exceeded.
hmat_rel_err = norm(H_full - M, 'fro') / norm(M, 'fro')
y = MM * x;
y1_rel_err = norm(y1 - y, 2) / norm(y, 2)
y2_rel_err = norm(y2 - 0.5 * y, 2) / norm(0.5 * y, 2)
