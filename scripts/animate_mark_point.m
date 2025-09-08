## Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function animate_mark_point(h, points, point_indices, pause_time)
  current_axes = get(h, "currentaxes");

  ## Clear all the labels first.
  clear_point(h, 0, current_axes);
  pause(pause_time)
  
  dim = size(points, 2);
  for m = 1:length(point_indices)
    switch(dim)
      case 2
	mark_point(h, current_axes, point_indices(m), points(:, 1), points(:, 2), []);
      case 3
	mark_point(h, current_axes, point_indices(m), points(:, 1), points(:, 2), points(:, 3));
    endswitch
    ## Update the text box to show the current marked index.
    set(get(get(h, 'children')(1), 'children')(3), 'string', num2str(point_indices(m)));
    
    pause(pause_time);

    clear_point(h, 0, current_axes);

    pause(pause_time);
  endfor
endfunction
