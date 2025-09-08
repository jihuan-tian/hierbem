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
load lapack-matrix-rsvd-degenerate-cases.output;

norm(A1 - U1 * diag(Sigma_r1) * VT1, "fro") / norm(A1, "fro")
norm(A2 - U2 * diag(Sigma_r2) * VT2, "fro") / norm(A2, "fro")
norm(A3 - U3 * diag(Sigma_r3) * VT3, "fro") / norm(A3, "fro")
