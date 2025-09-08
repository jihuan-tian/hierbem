## Copyright (C) 2021-2023 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

## Generate symmetric matrix.
A = [-1, 5,  2,  -3, 6,  1,  -2, 4; 2,  -3, -4, 1,  -3, -1, 1,  2;  -2, 4,  2,  -1, 3,  1,  -1, 3;  -3, 7, 2,  -3, 7,  2,  -2, 2;  1,  0,  0,  -1, 1,  -4, 0, 0;  0,  2,  0,  -2, 3,  -1, -1, 6;  -2, 4,  3,  -2, 4,  -1, -1, 3;  3,  -4, -6, 1,  -3, -3, 1,  -2];
M = tril(A * A');
save("-text", "M.dat", "M");
