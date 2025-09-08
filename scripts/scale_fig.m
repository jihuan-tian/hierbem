## Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function scale_fig(h, scale_factor)
  ## scale_fig - This function should be called after the plotting is finished.
  
  switch (length(scale_factor))
    case 1
      ## Scale the figure with a same factor for width and height.
      set_fig_size(h, get_fig_size(h) * scale_factor);
    case 2
      ## Scale the figure with different factors for width and height.
      set_fig_size(h, get_fig_size(h) .* scale_factor);
  endswitch
endfunction
