function mark_point_in_point_box(h, evt, current_axes, point_box, x, y, z)
  hold(current_axes, "on");

  point_index_str = get(point_box, "string");
  point_index = eval(cstrcat("[", point_index_str, "]"));

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
