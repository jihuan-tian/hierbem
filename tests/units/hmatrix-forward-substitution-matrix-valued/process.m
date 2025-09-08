## Copyright (C) 2021-2023 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

clear -x enable_figure;

load "L.dat";
load "Z.dat";
load "hmatrix-forward-substitution-matrix-valued.output";

L_rel_err = norm(L_full - L, 'fro') / norm(L, 'fro')
Z_rel_err = norm(Z_full - Z, 'fro') / norm(Z, 'fro')

X_octave = L \ Z;
X_rel_err = norm(X - X_octave, 'fro') / norm(X_octave, 'fro')
