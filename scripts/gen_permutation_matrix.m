## Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function sp = gen_permutation_matrix(permutation)
  original_indices = reshape(1:length(permutation), size(permutation, 1), size(permutation, 2));
  values = ones(size(permutation));
  sp = sparse(permutation, original_indices, values);
endfunction
