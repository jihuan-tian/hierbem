## Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function x_permuted = permute_vector(x, permutation)
  x_permuted = zeros(size(x));

  for m = 1:length(x_permuted)
    x_permuted(m) = x(permutation(m));
  endfor
endfunction
