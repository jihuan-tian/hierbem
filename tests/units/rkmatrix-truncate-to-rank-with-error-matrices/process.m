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
load "rkmatrix-truncate-to-rank-with-error-matrices.output";

## rank(M) is 2.
rank(M)
norm(M - A.A * A.B', "fro") / norm(M, "fro")
norm(M - A_trunc_to_1.A * A_trunc_to_1.B', "fro") / norm(M, "fro")
norm(M - A_trunc_to_2.A * A_trunc_to_2.B', "fro") / norm(M, "fro")
norm(M - A_trunc_to_3.A * A_trunc_to_3.B', "fro") / norm(M, "fro")

## Check the sum of A*B^T and C*D^T when the truncation rank is 1.
norm(M - (A_trunc_to_1.A * A_trunc_to_1.B' + C_trunc_to_1 * D_trunc_to_1'), 'fro') / norm(M, 'fro')
