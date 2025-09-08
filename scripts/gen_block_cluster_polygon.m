## Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function block_polygon = gen_block_cluster_polygon(tau_range, sigma_range, unit_size)
  yrange = [(tau_range(1) - 0.5) * unit_size, (tau_range(2) - 1 + 0.5) * unit_size] + 1;
  xrange = [(sigma_range(1) - 0.5) * unit_size, (sigma_range(2) - 1 + 0.5) * unit_size] + 1;

  block_polygon = [xrange(1), yrange(1);
		   xrange(1), yrange(2);
		   xrange(2), yrange(2);
		   xrange(2), yrange(1)];
endfunction
