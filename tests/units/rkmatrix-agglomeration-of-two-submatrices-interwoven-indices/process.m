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
load rkmatrix-agglomeration-of-two-submatrices-interwoven-indices.output;

norm(M_agglomerated1 - M_agglomerated1_rk.A * M_agglomerated1_rk.B', 'fro') / norm(M_agglomerated1, 'fro')
norm(M_agglomerated2 - M_agglomerated2_rk.A * M_agglomerated2_rk.B', 'fro') / norm(M_agglomerated2, 'fro')
