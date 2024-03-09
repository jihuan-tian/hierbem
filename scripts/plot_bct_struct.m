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
  p.addParameter("connect_blocks", false, @isbool);
  p.addParameter("connection_arrow_length", 0.5, val_float);
  p.addParameter("connection_arrow_width", 0.25, val_float);
  p.addParameter("connection_arrow_type", 1, val_float);

  p.parse(varargin{2:end});

  block_clusters = read_bct(filename);

  hold on;
  axis off;

  number_of_blocks = length(block_clusters);
  block_list = cell(number_of_blocks, 1);
  for m = 1:number_of_blocks
    yrange = [(block_clusters{m}.tau(1) - 0.5) * p.Results.unit_size, (block_clusters{m}.tau(end) - 1 + 0.5) * p.Results.unit_size] + 1;
    xrange = [(block_clusters{m}.sigma(1) - 0.5) * p.Results.unit_size, (block_clusters{m}.sigma(end) - 1 + 0.5) * p.Results.unit_size] + 1;
    xlength = xrange(2) - xrange(1);

    current_block_shape = [xrange(1), yrange(1);
		   xrange(1), yrange(2);
		   xrange(2), yrange(2);
		   xrange(2), yrange(1)];
    block_list{m} = current_block_shape;

    ## Draw the block.
    if (p.Results.fill_block)
      if (block_clusters{m}.is_near_field)
	block_color = p.Results.near_field_block_color;
      else
	block_color = p.Results.far_field_block_color;
      end
      
      fillPolygon(current_block_shape, block_color);
    endif
    
    ## Draw the block border.
    drawPolygon(current_block_shape, "color", p.Results.border_color, "linewidth", p.Results.border_width);

    if (isfield(block_clusters{m}, "rank") && p.Results.show_rank)
      ## Label the rank of the matrix block.
      text_x_coord = (xrange(2) + xrange(1)) / 2;
      text_y_coord = (yrange(2) + yrange(1)) / 2;

      if (block_clusters{m}.is_near_field)
	text(text_x_coord, text_y_coord, num2str(block_clusters{m}.rank), "fontsize", xlength / 2, "horizontalalignment", "center", "verticalalignment", "middle", "color", p.Results.near_field_rank_color);
      else
	text(text_x_coord, text_y_coord, num2str(block_clusters{m}.rank), "fontsize", xlength / 2, "horizontalalignment", "center", "verticalalignment", "middle", "color", p.Results.far_field_rank_color);
      endif
    endif
  endfor

  ## Connect successive blocks to show the traversal path.
  if (p.Results.connect_blocks)
    for m = 2:number_of_blocks
      plot_connecting_edge(block_list{m-1}, block_list{m}, p.Results.connection_arrow_length, p.Results.connection_arrow_width, p.Results.connection_arrow_type);
    endfor
  endif

  axis equal;
  set(gca, "ydir", "reverse");
endfunction
