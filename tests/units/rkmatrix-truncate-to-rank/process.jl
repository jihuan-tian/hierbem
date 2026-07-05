# Copyright (C) 2026 Jihuan Tian <jihuan_tian@hotmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. The full text of the license can be found in the
# file LICENSE at the top level directory of HierBEM.

using LinearAlgebra

A = [-8.847068200644447, -23.507360878883809, -38.167653557123103, -52.827946235362397, -67.488238913601691, -82.148531591840992]
B = [-0.365054522886896, -0.381924854601463, -0.398795186316030, -0.415665518030598, -0.432535849745165, -0.449406181459732]
A_rk_full_jl = A * adjoint(B)

A_complex = [1.267143553221514 + -1.218034284635582im, 1.408752864104520 + -1.256671551628080im, -0.195062280419648 + -0.819080029155473im, -2.069761632847959 + -0.307579344922377im, -2.491755254176841 + -0.192440869106438im, -1.073064155079812 + -0.579522385384543im]
B_complex = [0.216313558565074 + -0.000000000000000im, 0.431109014289334 + -0.154895866886818im, 0.559980937113453 + -0.247829528522745im, 0.484445075516260 + -0.193358204985249im, 0.273948752298739 + -0.041562579927325im, 0.122021316273549 + 0.067997149014163im]
A_complex_rk_full_jl = A_complex * adjoint(B_complex)
