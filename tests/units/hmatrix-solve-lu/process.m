## Copyright (C) 2021-2023 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

% Clear all vars except enable_figure
clear -x enable_figure;

load "hmatrix-solve-lu.output";
load "M.dat";
load "b.dat";

% Calculate relative error between H-Matrix and full matrix based on Frobenius-norm
hmat_rel_err = norm(H_full - M, 'fro') / norm(M, 'fro')

% Calculate relative error between H-matrix and full matrix solution based on 2-norm
x_octave = M \ b;
x_rel_err = norm(x_octave - x, 2) / norm(x_octave, 2)
