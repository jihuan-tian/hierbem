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
