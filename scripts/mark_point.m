## Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function mark_point(h, current_axes, point_index, x, y, z)
  hold(current_axes, "on");

  for l = 1:length(point_index)
    for m = 1:length(x)
      current_label = cellfun(@str2num, strsplit(num2str(m), ","), "UniformOutput", true);
      idx = find(current_label == point_index(l));

      if (!isempty(idx))
	if (isempty(z))
	  obj = plot(current_axes, x(m), y(m), "bo", "markersize", 30);
	  set(current_axes, "userdata", [get(current_axes, "userdata"), obj]);
	  obj = plot(current_axes, x(m), y(m), "b+", "markersize", 30);
	  set(current_axes, "userdata", [get(current_axes, "userdata"), obj]);
	else
	  obj = plot3(current_axes, x(m), y(m), z(m), "bo", "markersize", 30);
	  set(current_axes, "userdata", [get(current_axes, "userdata"), obj]);
	  obj = plot3(current_axes, x(m), y(m), z(m), "b+", "markersize", 30);
	  set(current_axes, "userdata", [get(current_axes, "userdata"), obj]);
	endif

	break;
      endif
    endfor
  endfor

  hold(current_axes, "off");
endfunction
