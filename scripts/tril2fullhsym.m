## Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function M_hsym = tril2fullhsym(L)
  ## tril2fullhsym - Convert a lower triangular matrix to a full
  ## Hermite symmetric matrix.

  [m, n] = size(L);
  if (m != n)
    error("Matrix row and column dimensions should be the same!");
  else
    ## Check if the imaginary parts of diagonal entries are zero.
    if sum(imag(diag(L))) != 0
      error("The diagonal entries of L should be real values.");
    else
      M_hsym = L + ctranspose(L - (eye(n) .* diag(L)));
    endif
  endif
endfunction
