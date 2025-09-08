# Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. The full text of the license can be found in the
# file LICENSE at the top level directory of HierBEM.

# Complex valued
A = reshape([sin(i) + cos(i)im for i in range(1.0, 15.0)], 3, 5)
B = reshape([cos(i) + sin(i)im for i in range(1.0, 15.0)], 3, 5)
C1 = [i/10. + (j/10.)im for i in range(1.0, 3.0), j in range(1.0, 3.0)]
C1 = C1 * adjoint(C1)
C2 = [i/10. + (j/10.)im for i in range(1.0, 5.0), j in range(1.0, 5.0)]
C2 = C2 * adjoint(C2)
alpha1 = 2.5 + 1.2im
alpha2 = 2.5

A_mul_AH = A * adjoint(A)
AH_mul_A = adjoint(A) * A
A_mul_BH = A * adjoint(B)
AH_mul_B = adjoint(A) * B

C1_add_alpha1_A_mul_AH = alpha1 * A * adjoint(A) + C1
C2_add_alpha1_AH_mul_A = alpha1 * adjoint(A) * A + C2
C1_add_alpha1_A_mul_BH = alpha1 * A * adjoint(B) + C1
C2_add_alpha1_AH_mul_B = alpha1 * adjoint(A) * B + C2

C1_add_alpha2_A_mul_AH = alpha2 * A * adjoint(A) + C1
C2_add_alpha2_AH_mul_A = alpha2 * adjoint(A) * A + C2
C1_add_alpha2_A_mul_BH = alpha2 * A * adjoint(B) + C1
C2_add_alpha2_AH_mul_B = alpha2 * adjoint(A) * B + C2
