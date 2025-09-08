# Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. The full text of the license can be found in the
# file LICENSE at the top level directory of HierBEM.

using LinearAlgebra

A = transpose(reshape(1.:15., 5, 3))
F = svd(A, full = true)
U = Matrix(F.U)
VT = Matrix(F.Vt)
Sigma_r = F.S

A_complex = [sin(i) + cos(j)im for i in range(1.0, 3.0), j in range(1.0, 5.0)]
F_complex = svd(A_complex, full = true)
U_complex = Matrix(F_complex.U)
VT_complex = Matrix(F_complex.Vt)
Sigma_r_complex = F_complex.S
