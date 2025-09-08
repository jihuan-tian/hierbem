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
load hmatrix-rkmatrix-mmult.output;

## Even though M2 has a rank 20, there are only two significant
## singular values.
rank(M2)
bar(svd(M2))
## The rank of the rank-k matrix is 2, which approximates M2 very
## well.
rank(M2_rk.A * M2_rk.B')
norm(M2 - M2_rk.A * M2_rk.B', 'fro') / norm(M2, 'fro')

M = M1 * M2;
norm(M - M_rk.A * M_rk.B', 'fro') / norm(M, 'fro')
