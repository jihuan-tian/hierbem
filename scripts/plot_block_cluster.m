function plot_block_cluster(bc, unit_size, enable_display_rank, is_fill)
  if (!exist("enable_display_rank", "var"))
    enable_display_rank = true;
  endif

  if (!exist("is_fill", "var"))
    is_fill = true;
  endif

  yrange = [(bc.tau(1) - 0.5) * unit_size, (bc.tau(end) - 1 + 0.5) * unit_size] + 1;
  xrange = [(bc.sigma(1) - 0.5) * unit_size, (bc.sigma(end) - 1 + 0.5) * unit_size] + 1;
  xlength = xrange(2) - xrange(1);

  block_shape = [xrange(1), yrange(1);
		 xrange(1), yrange(2);
		 xrange(2), yrange(2);
		 xrange(2), yrange(1)];

  if (bc.is_near_field)
    shape_color = "r";
  else
    shape_color = "g";
  end

  if (is_fill)
    fillPolygon(block_shape, shape_color);
  endif
  
  drawPolygon(block_shape, "k", "linewidth", 1);

  if (isfield(bc, "rank") && enable_display_rank)
    ## Label the rank of the matrix block.
    text_x_coord = (xrange(2) + xrange(1)) / 2;
    text_y_coord = (yrange(2) + yrange(1)) / 2;

    text(text_x_coord, text_y_coord, num2str(bc.rank), "fontsize", min(xlength * 4, xlength * 8), "horizontalalignment", "center", "verticalalignment", "middle");
  endif
endfunction
