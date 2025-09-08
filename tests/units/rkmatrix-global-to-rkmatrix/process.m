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
load rkmatrix-global-to-rkmatrix.output;

tau = [2, 3, 4, 5, 7, 10, 18, 19] + 1;
sigma = [3, 4, 8, 9, 11, 13, 15, 16, 17] + 1;
M_b_octave = M(tau, sigma);

norm(M - M_rk.A * M_rk.B', 'fro') / norm(M, 'fro')
norm(M_b_octave - M_b_rk.A * M_b_rk.B', 'fro') / norm(M_b_octave, 'fro')
norm(M_b_octave - M_b_rk1.A * M_b_rk1.B', 'fro') / norm(M_b_octave, 'fro')
