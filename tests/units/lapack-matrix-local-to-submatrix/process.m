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
load lapack-matrix-local-to-submatrix.output;

tau = (5:12) + 1;
sigma = (7:14) + 1;
M_b_octave = M(tau, sigma);

tau_subset = (7:10) + 1;
sigma_subset = (10:12) + 1;
M_b_submatrix_octave = M(tau_subset, sigma_subset);

norm(M_b_submatrix - M_b_submatrix_octave, 'fro') / norm(M_b_submatrix, 'fro')
