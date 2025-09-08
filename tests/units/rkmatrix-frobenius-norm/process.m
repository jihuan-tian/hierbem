## Copyright (C) 2022-2023 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

clear all;
load "rkmatrix-frobenius-norm.output";

rkmat = A * B';
norm(rkmat, 'fro');
calculated_value = 7.750767182503684e+04;
(norm(rkmat, 'fro') - calculated_value) / norm(rkmat, 'fro')
