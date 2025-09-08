## Copyright (C) 2021-2024 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

clear all;

load_packages;
load hmatrix-solve-cholesky.output;
load M.dat;
load b.dat;

## Restore the matrix M to be symmetric.
MM = tril2fullsym(M);

## Calculate relative error between H-Matrix and full matrix based on
## Frobenius-norm. Only the lower triangular part is compared.
hmat_rel_err = norm(H_full - M, 'fro') / norm(M, 'fro')

## Calculate the relative error between L*L^T and the original symmetric matrix.
product_hmat_rel_err = norm(L_full*L_full' - MM, 'fro') / norm(MM, 'fro')

## Calculate the relative error between H-matrix and full matrix solution
## vectors.
x_octave = MM \ b;
x_rel_err = norm(x_octave - x, 2) / norm(x_octave, 2)
