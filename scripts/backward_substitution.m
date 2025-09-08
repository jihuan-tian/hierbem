## Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function y = backward_substitution(U, b)
  y = b;

  for m = size(U, 1):-1:1
    for n = (m+1):size(U,1)
      y(m) = y(m) - U(m,n) * y(n);
    endfor
    y(m) = y(m) / U(m,m);
  endfor
endfunction
