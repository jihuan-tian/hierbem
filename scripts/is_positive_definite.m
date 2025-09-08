## Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function ret = is_positive_definite(M)
  lambda  = eig(M);
  min_lambda = min(lambda);
  if (min_lambda < 0)
    fprintf(stdout(),
	    "The matrix is not positive definitive with the minimum eigen value %g!\n", min_lambda);

    ret = false;
  else
    fprintf(stdout(),
	    "The matrix is positive definitive with the minimum eigen value %g!\n", min_lambda);

    ret = true;
  endif
endfunction
