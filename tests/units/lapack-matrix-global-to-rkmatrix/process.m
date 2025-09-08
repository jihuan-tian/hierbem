## Copyright (C) 2021-2023 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

clear all;
format long;
load lapack-matrix-global-to-rkmatrix.output;

tau = [7,8,9,10] + 1;
sigma = [9,10,11,12] + 1;
M_b = M(tau, sigma);

norm(M_b - rkmat_no_trunc.A * rkmat_no_trunc.B', 'fro') / norm(M_b, 'fro')
norm(M_b - rk1mat.A * rk1mat.B', 'fro') / norm(M_b, 'fro')
