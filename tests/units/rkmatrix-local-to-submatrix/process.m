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
load rkmatrix-local-to-submatrix.output;

tau_subset = [3, 7, 10, 19] + 1;
sigma_subset = [8, 13, 17] + 1;
M_b_submatrix_octave = M(tau_subset, sigma_subset);

norm(M_b_submatrix_octave - M_b_submatrix, 'fro') / norm(M_b_submatrix_octave)
