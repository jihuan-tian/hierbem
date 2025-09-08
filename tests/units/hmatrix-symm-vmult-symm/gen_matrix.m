## Copyright (C) 2021-2024 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

## Generate the lower triangular part of the symmetric positive definite matrix.
p = 6;
n = 2^p;
M = randn(n);
M = M * M';
M = tril(M);
x = randn(n, 1);
save("-text", "M.dat", "M");
save("-text", "x.dat", "x");
