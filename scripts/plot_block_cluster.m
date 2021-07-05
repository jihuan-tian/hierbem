function plot_block_cluster(bc, unit_size)
  yrange = [(bc.tau(1) - 0.5) * unit_size, (bc.tau(end) + 0.5) * unit_size];
  xrange = [(bc.sigma(1) - 0.5) * unit_size, (bc.sigma(end) + 0.5) * unit_size];

  block_shape = [xrange(1), yrange(1);
		 xrange(1), yrange(2);
		 xrange(2), yrange(2);
		 xrange(2), yrange(1)];

  if (bc.is_near_field)
    shape_color = "r";
  else
    shape_color = "g";
  end

  fillPolygon(block_shape, shape_color);
  drawPolygon(block_shape, "k", "linewidth", 1);

  if (isfield(bc, "rank"))
    ## Label the rank of the matrix block.
    text_x_coord = (xrange(2) + xrange(1)) / 2;
    text_y_coord = (yrange(2) + yrange(1)) / 2;

    text(text_x_coord, text_y_coord, num2str(bc.rank), "fontsize", 8);
  endif
endfunction
