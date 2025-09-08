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

n = 16 * 1024
v1 = rand(n, 1)
v2 = rand(n, 1)
v1_complex = rand(n, 1) + rand(n, 1)im
v2_complex = rand(n, 1) + rand(n, 1)im

inner_product1 = dot(v1, v2)
inner_product2 = dot(v2_complex, v1_complex)
inner_product3 = dot(v2, v1_complex)

linear_combination1 = sum(v1 .* v2)
linear_combination2 = sum(v1_complex .* v2_complex)
linear_combination3 = sum(v1_complex .* v2)
