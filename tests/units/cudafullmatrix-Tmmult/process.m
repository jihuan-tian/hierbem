## Copyright (C) 2023 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

clear all;

A = reshape([1, 3, 5, 7, 9, 10], 3, 2);
B = reshape([2, 8, 9, 7, 1, 3, 11, 20, 13], 3, 3);
C = A' * B
C = reshape([1, 1, 1, 2, 2, 2], 2, 3);
C = C + A' * B
