## Copyright (C) 2023 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

function gen_matrix(trial_no)
  p = 6;
  n = 2^p;
  M = randn(n);
  M = M * M';
  b = randn(n, 1);
  
  save("-text", cstrcat("M", num2str(trial_no), ".dat"), "M");
  save("-text", cstrcat("b", num2str(trial_no), ".dat"), "b");
endfunction
