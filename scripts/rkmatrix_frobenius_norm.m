## Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function result = rkmatrix_frobenius_norm(A, B)
  k = size(A, 2);
  if (k != size(B, 2))
    error("Matrix A and B should have the same number of columns");
  endif

  result = 0;
  
  for m = 1:k
    for n = 1:k
      result = result + sum(A(:,m) .* conj(A(:,n))) * sum(B(:,m) .* conj(B(:,n)));
    endfor
  endfor
endfunction
