## Copyright (C) 2021-2025 Jihuan Tian <jihuan_tian@hotmail.com>
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
M = tril(M * transpose(M));
x = randn(n, 1);
y0 = randn(n, 1);
M_complex = complex(randn(n), randn(n));
M_complex = tril(M_complex * transpose(M_complex));
x_complex = complex(randn(n, 1), randn(n, 1));
y0_complex = complex(randn(n, 1), randn(n, 1));
save("-text", "M.dat", "M", "M_complex");
save("-text", "xy.dat", "x", "y0", "x_complex", "y0_complex");
