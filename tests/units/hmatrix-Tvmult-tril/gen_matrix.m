## Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

## Generate a lower triangular matrix.
p = 6;
n = 2^p;
M = tril(randn(n));
x = randn(n, 1);
M_complex = tril(complex(randn(n), randn(n)));
x_complex = complex(randn(n, 1), randn(n, 1));
save("-text", "M.dat", "M", "M_complex");
save("-text", "x.dat", "x", "x_complex");
