# Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. The full text of the license can be found in the
# file LICENSE at the top level directory of HierBEM.

a = UInt32(10)
b = Int32(-5)
c = Float32(3.5)
d = Float64(-3.5)
e = ComplexF32(1.2 + 0.5im)
f = ComplexF64(-1.2 - 0.5im)
g = collect(Float32, range(1.0, 5.0, step = 0.2))
h = collect(Float64, range(5.0, 1.0, step = -0.2))
i = [ComplexF32(x + (x + 1)im) for x in range(1.0, 5.0, step = 0.2)]
j = [ComplexF64(x + (x + 1)im) for x in range(5.0, 1.0, step = -0.2)]
k = reshape(h, 3, 7)
