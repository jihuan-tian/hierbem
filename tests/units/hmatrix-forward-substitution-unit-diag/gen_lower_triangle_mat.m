## Copyright (C) 2021-2023 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

p = 5;
n = 2^p;
M = randn(n);
M = M * M';
## Scale the matrix and make the maximum diagonal entry be 1.
M = M / max(diag(M));
## Get the lower triangular part of the matrix and enforce all diagonal entries
## to be 1.
L = tril(M);
for m = 1:n
  L(m,m) =1.0;
endfor
save("-text", "L.dat", "L");

b = rand(n, 1);
save("-text", "b.dat", "b");
