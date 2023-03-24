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
