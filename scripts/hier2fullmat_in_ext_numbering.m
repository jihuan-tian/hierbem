## Copyright (C) 2026 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function fullmat_in_ext_numbering = hier2fullmat_in_ext_numbering(fullmat_in_int_numbering, row_i2e, col_i2e)
  ## hier2fullmat_in_ext_numbering - Convert a matrix from internal DoF
  ## numbering to external DoF numbering by permuting rows and columns.

  if iscomplex(fullmat_in_int_numbering)
    fullmat_in_ext_numbering = complex(zeros(max(row_i2e)+1, max(col_i2e)+1));
  else
    fullmat_in_ext_numbering = zeros(max(row_i2e)+1, max(col_i2e)+1);
  endif

  fullmat_in_ext_numbering_tmp = fullmat_in_ext_numbering;
  ## Permute rows.
  for m = 1:size(fullmat_in_int_numbering, 1)
    fullmat_in_ext_numbering_tmp(row_i2e(m)+1, :) = fullmat_in_int_numbering(m, :);
  endfor

  ## Permute columns.
  for n = 1:size(fullmat_in_int_numbering, 2)
    fullmat_in_ext_numbering(:, col_i2e(n)+1) = fullmat_in_ext_numbering_tmp(:, n);
  endfor
endfunction
