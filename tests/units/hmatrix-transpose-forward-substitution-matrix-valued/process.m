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
load "U.dat";
load "Z.dat";
load "hmatrix-transpose-forward-substitution-matrix-valued.output";

U_rel_err = norm(U_full - U, 'fro') / norm(U, 'fro')
Z_rel_err = norm(Z_full - Z, 'fro') / norm(Z, 'fro')

X_octave = (U' \ Z')';
X_rel_err = norm(X - X_octave, 'fro') / norm(X_octave, 'fro')
