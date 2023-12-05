function plot_bct_struct(varargin)
  ## filename, enable_display_rank, is_fill

  ## The first argument is the file name storing the block cluster tree
  ## structure.
  filename = varargin{1};

  ## Value checkers
  val_float = @(x) isscalar(x) && isfloat(x);

  ## Use input parser to handle remaining parameters.
  p = inputParser();
  p.FunctionName = "plot_bct_struct";
  p.addParameter("unit_size", 1, val_float);
  p.addParameter("show_rank", true, @isbool);
  p.addParameter("fill_block", true, @isbool);
  p.addParameter("near_field_block_color", "r", @ischar);
  p.addParameter("far_field_block_color", "g", @ischar);
  p.addParameter("border_color", "white", @ischar);
  p.addParameter("border_width", 1, val_float);
  p.addParameter("near_field_rank_color", "k", @ischar);
  p.addParameter("far_field_rank_color", "k", @ischar);

  p.parse(varargin{2:end});

  block_clusters = read_bct(filename);

  hold on;
  axis off;
  
  for m = 1:length(block_clusters)
    yrange = [(block_clusters{m}.tau(1) - 0.5) * p.Results.unit_size, (block_clusters{m}.tau(end) - 1 + 0.5) * p.Results.unit_size] + 1;
    xrange = [(block_clusters{m}.sigma(1) - 0.5) * p.Results.unit_size, (block_clusters{m}.sigma(end) - 1 + 0.5) * p.Results.unit_size] + 1;
    xlength = xrange(2) - xrange(1);

    block_shape = [xrange(1), yrange(1);
		   xrange(1), yrange(2);
		   xrange(2), yrange(2);
		   xrange(2), yrange(1)];

    ## Draw the block.
    if (p.Results.fill_block)
      if (block_clusters{m}.is_near_field)
	block_color = p.Results.near_field_block_color;
      else
	block_color = p.Results.far_field_block_color;
      end
      
      fillPolygon(block_shape, block_color);
    endif
    
    ## Draw the block border.
    drawPolygon(block_shape, "color", p.Results.border_color, "linewidth", p.Results.border_width);

    if (isfield(block_clusters{m}, "rank") && p.Results.show_rank)
      ## Label the rank of the matrix block.
      text_x_coord = (xrange(2) + xrange(1)) / 2;
      text_y_coord = (yrange(2) + yrange(1)) / 2;

      if (block_clusters{m}.is_near_field)
	text(text_x_coord, text_y_coord, num2str(block_clusters{m}.rank), "fontsize", min(xlength * 4, xlength * 8), "horizontalalignment", "center", "verticalalignment", "middle", "color", p.Results.near_field_rank_color);
      else
	text(text_x_coord, text_y_coord, num2str(block_clusters{m}.rank), "fontsize", min(xlength * 4, xlength * 8), "horizontalalignment", "center", "verticalalignment", "middle", "color", p.Results.far_field_rank_color);
      endif
    endif
  endfor

  axis equal;
  set(gca, "ydir", "reverse");
endfunction
