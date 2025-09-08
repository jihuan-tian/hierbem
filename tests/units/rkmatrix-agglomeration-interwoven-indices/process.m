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
load rkmatrix-agglomeration-interwoven-indices.output;

norm(M11 - M11_rk.A * M11_rk.B', 'fro') / norm(M11, 'fro')
norm(M12 - M12_rk.A * M12_rk.B', 'fro') / norm(M12, 'fro')
norm(M21 - M21_rk.A * M21_rk.B', 'fro') / norm(M21, 'fro')
norm(M22 - M22_rk.A * M22_rk.B', 'fro') / norm(M22, 'fro')
norm(M - M_rk.A * M_rk.B', 'fro') / norm(M, 'fro')
