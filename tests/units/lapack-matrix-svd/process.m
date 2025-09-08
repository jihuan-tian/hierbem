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
load "lapack-matrix-svd.output";

m = 3;
n = 5;

## Construct the Sigma matrix from Sigma_r.
Sigma = [diag(Sigma_r1), zeros(m, n - m)];
norm(A - U1 * Sigma * VT1, "fro") / norm(A, "fro")

Sigma = [diag(Sigma_r2), zeros(m, n - m)];
norm(A - U2 * Sigma * VT2, "fro") / norm(A, "fro")

Sigma = [diag(Sigma_r3), zeros(m, n - m)];
norm(A - U3 * Sigma * VT3, "fro") / norm(A, "fro")

Sigma = [diag(Sigma_r4), zeros(m, n - m)];
norm(A - U4 * Sigma * VT4, "fro") / norm(A, "fro")

Sigma = [diag(Sigma_r5), zeros(m, n - m)];
norm(A - U5 * Sigma * VT5, "fro") / norm(A, "fro")
