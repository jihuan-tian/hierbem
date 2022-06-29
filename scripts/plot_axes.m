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
