## Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function plot_task_edges(varargin)
  bct_file = varargin{1};
  task_edges_file = varargin{2};

  ## Value checkers
  val_float = @(x) isscalar(x) && isfloat(x);

  ## Color checkers
  color_triple = @(x) (length(x) == 3 && isfloat(x)) || (length(x) == 1 && ischar(x));

  ## Use input parser to handle remaining parameters.
  p = inputParser();
  p.FunctionName = "plot_task_edges";
  p.addParameter("unit_size", 1, val_float);
  p.addParameter("border_color", "black", @ischar);
  p.addParameter("border_width", 0.25, val_float);
  p.addParameter("update_mark_border_color", [255,128,0]/255, @color_triple);
  p.addParameter("update_mark_border_width", 1, val_float);
  p.addParameter("solve_mark_border_color", 'r', @color_triple);
  p.addParameter("solve_mark_border_width", 1, val_float);
  p.addParameter("factorize_mark_border_color", 'r', @color_triple);
  p.addParameter("factorize_mark_border_width", 1, val_float);
  p.addParameter("factorize_color", [250,130,120]/255, @color_triple);
  p.addParameter("solve_color", [250,188,120]/255, @color_triple);
  p.addParameter("update_color", [139,219,236]/255, @color_triple);
  p.addParameter("arrow_length", 1, val_float);
  p.addParameter("arrow_width", 0.5, val_float);
  p.addParameter("arrow_type", 1, val_float);

  p.parse(varargin{3:end});

  hold on;
  
  fid = fopen(task_edges_file, 'r');

  while (true)
    line_str = fgetl(fid);

    if (line_str == -1)
      break;
    else
      fields = strsplit(line_str, ':');
      edge_type_and_level_info = fields{1};
      edge_type_and_level_info_fields = strsplit(edge_type_and_level_info, 'at level');
      edge_type = strtrim(edge_type_and_level_info_fields{1});
      level_info = strtrim(edge_type_and_level_info_fields{2});

      switch (edge_type)
	case {'factorize-to-solve-lower', 'factorize-to-solve-upper'}
	  clusters = strsplit(strtrim(fields{2}), {', ', '-->'});
	  factorize_block_tau = eval(strrep(strtrim(clusters{1}), ')', ']'));
	  factorize_block_sigma = eval(strrep(strtrim(clusters{2}), ')', ']'));
	  solve_block_tau = eval(strrep(strtrim(clusters{3}), ')', ']'));
	  solve_block_sigma = eval(strrep(strtrim(clusters{4}), ')', ']'));

	  factorize_block = gen_block_cluster_polygon(factorize_block_tau, factorize_block_sigma, p.Results.unit_size);
	  fillPolygon(factorize_block, p.Results.factorize_color);
	  drawPolygon(factorize_block, 'color', p.Results.factorize_mark_border_color, 'linewidth', p.Results.factorize_mark_border_width);
	  
	  solve_block = gen_block_cluster_polygon(solve_block_tau, solve_block_sigma, p.Results.unit_size);
	  fillPolygon(solve_block, p.Results.solve_color);
	  drawPolygon(solve_block, 'color', p.Results.solve_mark_border_color, 'linewidth', p.Results.solve_mark_border_width);
	case {'solve-lower-to-update', 'solve-upper-to-update'}
	  clusters = strsplit(strtrim(fields{2}), {', ', '-->'});
	  solve_block_tau = eval(strrep(strtrim(clusters{1}), ')', ']'));
	  solve_block_sigma = eval(strrep(strtrim(clusters{2}), ')', ']'));
	  update_block_tau = eval(strrep(strtrim(clusters{3}), ')', ']'));
	  update_block_sigma = eval(strrep(strtrim(clusters{4}), ')', ']'));

	  solve_block = gen_block_cluster_polygon(solve_block_tau, solve_block_sigma, p.Results.unit_size);
	  fillPolygon(solve_block, p.Results.solve_color);
	  drawPolygon(solve_block, 'color', p.Results.solve_mark_border_color, 'linewidth', p.Results.solve_mark_border_width);
	  
	  update_block = gen_block_cluster_polygon(update_block_tau, update_block_sigma, p.Results.unit_size);
	  fillPolygon(update_block, p.Results.update_color);
	  drawPolygon(update_block, 'color', p.Results.update_mark_border_color, 'linewidth', p.Results.update_mark_border_width);
      endswitch
    endif
  endwhile

  ## Reset the file seek pointer and restart reading file for plotting
  ## connection edges.
  fseek(fid, 0);

  while (true)
    line_str = fgetl(fid);

    if (line_str == -1)
      break;
    else
      fields = strsplit(line_str, ':');
      edge_type_and_level_info = fields{1};
      edge_type_and_level_info_fields = strsplit(edge_type_and_level_info, 'at level');
      edge_type = strtrim(edge_type_and_level_info_fields{1});
      level_info = strtrim(edge_type_and_level_info_fields{2});

      switch (edge_type)
	case {'factorize-to-solve-lower', 'factorize-to-solve-upper'}
	  clusters = strsplit(strtrim(fields{2}), {', ', '-->'});
	  factorize_block_tau = eval(strrep(strtrim(clusters{1}), ')', ']'));
	  factorize_block_sigma = eval(strrep(strtrim(clusters{2}), ')', ']'));
	  solve_block_tau = eval(strrep(strtrim(clusters{3}), ')', ']'));
	  solve_block_sigma = eval(strrep(strtrim(clusters{4}), ')', ']'));

	  factorize_block = gen_block_cluster_polygon(factorize_block_tau, factorize_block_sigma, p.Results.unit_size);
	  solve_block = gen_block_cluster_polygon(solve_block_tau, solve_block_sigma, p.Results.unit_size);

	  ## Connect the two blocks
	  plot_connecting_edge(factorize_block, solve_block, p.Results.arrow_length * p.Results.unit_size, p.Results.arrow_width * p.Results.unit_size, p.Results.arrow_type);
	case {'solve-lower-to-update', 'solve-upper-to-update'}
	  clusters = strsplit(strtrim(fields{2}), {', ', '-->'});
	  solve_block_tau = eval(strrep(strtrim(clusters{1}), ')', ']'));
	  solve_block_sigma = eval(strrep(strtrim(clusters{2}), ')', ']'));
	  update_block_tau = eval(strrep(strtrim(clusters{3}), ')', ']'));
	  update_block_sigma = eval(strrep(strtrim(clusters{4}), ')', ']'));

	  solve_block = gen_block_cluster_polygon(solve_block_tau, solve_block_sigma, p.Results.unit_size);
	  update_block = gen_block_cluster_polygon(update_block_tau, update_block_sigma, p.Results.unit_size);

	  ## Connect the two blocks
	  plot_connecting_edge(solve_block, update_block, p.Results.arrow_length * p.Results.unit_size, p.Results.arrow_width * p.Results.unit_size, p.Results.arrow_type);
      endswitch
    endif
  endwhile

  ## Plot the block cluster structure as the background.
  plot_bct_struct(bct_file, 'unit_size', p.Results.unit_size, 'border_color', p.Results.border_color, 'border_width', p.Results.border_width, 'show_rank', false, 'fill_block', false);

  fclose(fid);
endfunction
