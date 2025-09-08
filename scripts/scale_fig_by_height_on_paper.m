## Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function scale_fig_by_height_on_paper(h, new_height_cm)
  new_fig_size_inch = calc_fig_size_by_height_on_paper(h, new_height_cm);
  set_fig_size_on_paper(h, new_fig_size_inch);
endfunction
