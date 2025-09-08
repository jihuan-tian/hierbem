## Copyright (C) 2022-2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function plot_axes(origin, axis_length)
  ## plot_axes - Plot axes in 2D or 3D
  
  dimension = length(origin);

  hold on;
  switch(dimension)
    case 2
      plot([origin(1), origin(1) + axis_length], [origin(2), origin(2)], "r-");
      plot([origin(1), origin(1)], [origin(2), origin(2) + axis_length], "g-");
    case 3
      plot3([origin(1), origin(1) + axis_length], [origin(2), origin(2)], [origin(3), origin(3)], "r-");
      plot3([origin(1), origin(1)], [origin(2), origin(2) + axis_length], [origin(3), origin(3)], "g-");
      plot3([origin(1), origin(1)], [origin(2), origin(2)], [origin(3), origin(3) + axis_length], "b-");
    otherwise
      error(sprintf("Dimension %d is not implemented!", dimension));
  endswitch
endfunction
