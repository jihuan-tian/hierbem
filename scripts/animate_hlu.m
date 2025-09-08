## Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function animate_hlu(varargin)
  bct_file = varargin{1};
  lu_file = varargin{2};

  ## Value checkers
  val_float = @(x) isscalar(x) && isfloat(x);

  ## Color checkers
  color_triple = @(x) (length(x) == 3 && isfloat(x)) || (length(x) == 1 && ischar(x));

  ## Use input parser to handle remaining parameters.
  p = inputParser();
  p.FunctionName = "animate_hlu";
  p.addParameter("unit_size", 1, val_float);
  p.addParameter("border_color", "black", @ischar);
  p.addParameter("border_width", 1, val_float);
  p.addParameter("update_mark_border_color", [255,128,0]/255, @color_triple);
  p.addParameter("update_mark_border_width", 2, val_float);
  p.addParameter("diag_mark_border_color", 'r', @color_triple);
  p.addParameter("diag_mark_border_width", 2, val_float);
  p.addParameter("focus_mark_border_color", 'r', @color_triple);
  p.addParameter("focus_mark_border_width", 2, val_float);
  p.addParameter("factorize_color", [250,130,120]/255, @color_triple);
  p.addParameter("solve_color", [250,188,120]/255, @color_triple);
  p.addParameter("update_color", [139,219,236]/255, @color_triple);
  p.addParameter("time_interval", 1, val_float);

  p.parse(varargin{3:end});

  ## Plot the block cluster structure as the background.
  plot_bct_struct(bct_file, 'unit_size', p.Results.unit_size, 'border_color', p.Results.border_color, 'border_width', p.Results.border_width, 'show_rank', false, 'fill_block', false);
  hold on;
  
  fid = fopen(lu_file, 'r');

  counter = 1;
  while (true)
    line_str = fgetl(fid);

    if (line_str == -1)
      break;
    else
      fields = strsplit(line_str, ':');
      task_type = fields{1};

      switch (task_type)
	case 'lu_factorize'
	  title(cstrcat(num2str(counter), ': lu_factorize'), 'interpreter', 'none');
	  
	  bct = strsplit(strtrim(fields{2}), ', ');
	  tau_range = eval(strrep(strtrim(bct{1}), ')', ']'));
	  sigma_range = eval(strrep(strtrim(bct{2}), ')', ']'));

	  current_block = gen_block_cluster_polygon(tau_range, sigma_range, p.Results.unit_size);
	  fillPolygon(current_block, p.Results.factorize_color);
	  drawPolygon(current_block, 'color', p.Results.border_color, 'linewidth', p.Results.border_width);
	  marker = drawPolygon(current_block, 'color', p.Results.focus_mark_border_color, 'linewidth', p.Results.focus_mark_border_width);
	  pause(p.Results.time_interval);

	  delete(marker);

          pause(p.Results.time_interval);
	case 'lu_solve_upper'
	  title(cstrcat(num2str(counter), ': lu_solve_upper'), 'interpreter', 'none');
	  
	  bct = strsplit(strtrim(fields{2}), ', ');
	  tau_range = eval(strrep(strtrim(bct{1}), ')', ']'));
	  sigma_range = eval(strrep(strtrim(bct{2}), ')', ']'));

	  current_block = gen_block_cluster_polygon(tau_range, sigma_range, p.Results.unit_size);
	  fillPolygon(current_block, p.Results.solve_color);
	  drawPolygon(current_block, 'color', p.Results.border_color, 'linewidth', p.Results.border_width);
	  marker = drawPolygon(current_block, 'color', p.Results.focus_mark_border_color, 'linewidth', p.Results.focus_mark_border_width);
	  pause(p.Results.time_interval);

	  delete(marker);
	  
          pause(p.Results.time_interval);
	case 'lu_solve_lower'
	  title(cstrcat(num2str(counter), ': lu_solve_lower'), 'interpreter', 'none');
	  
	  bct = strsplit(strtrim(fields{2}), ', ');
	  tau_range = eval(strrep(strtrim(bct{1}), ')', ']'));
	  sigma_range = eval(strrep(strtrim(bct{2}), ')', ']'));

	  current_block = gen_block_cluster_polygon(tau_range, sigma_range, p.Results.unit_size);
	  fillPolygon(current_block, p.Results.solve_color);
	  drawPolygon(current_block, 'color', p.Results.border_color, 'linewidth', p.Results.border_width);
	  marker = drawPolygon(current_block, 'color', p.Results.focus_mark_border_color, 'linewidth', p.Results.focus_mark_border_width);
	  pause(p.Results.time_interval);

	  delete(marker);
	  
          pause(p.Results.time_interval);
	case 'lu_update'
	  title(cstrcat(num2str(counter), ': lu_update'), 'interpreter', 'none');
	  
	  bct = strsplit(strtrim(fields{2}), {', ', '*', '-->'});
	  L_tau_range = eval(strrep(strtrim(bct{1}), ')', ']'));
	  L_sigma_range = eval(strrep(strtrim(bct{2}), ')', ']'));
	  U_tau_range = eval(strrep(strtrim(bct{3}), ')', ']'));
	  U_sigma_range = eval(strrep(strtrim(bct{4}), ')', ']'));
	  tau_range = eval(strrep(strtrim(bct{5}), ')', ']'));
	  sigma_range = eval(strrep(strtrim(bct{6}), ')', ']'));

	  marker1 = drawPolygon(gen_block_cluster_polygon(L_tau_range, L_sigma_range, p.Results.unit_size), 'color', p.Results.update_mark_border_color, 'linewidth', p.Results.update_mark_border_width);
	  marker2 = drawPolygon(gen_block_cluster_polygon(U_tau_range, U_sigma_range, p.Results.unit_size), 'color', p.Results.update_mark_border_color, 'linewidth', p.Results.update_mark_border_width);
	  marker3 = drawPolygon(gen_block_cluster_polygon(U_tau_range, L_sigma_range, p.Results.unit_size), 'color', p.Results.diag_mark_border_color, 'linewidth', p.Results.diag_mark_border_width);

	  pause(p.Results.time_interval);

	  current_block = gen_block_cluster_polygon(tau_range, sigma_range, p.Results.unit_size);
	  fillPolygon(current_block, p.Results.update_color);
	  drawPolygon(current_block, 'color', p.Results.border_color, 'linewidth', p.Results.border_width);
	  marker = drawPolygon(current_block, 'color', p.Results.focus_mark_border_color, 'linewidth', p.Results.focus_mark_border_width);
	  
          pause(p.Results.time_interval);

	  delete(marker1);
	  delete(marker2);
	  delete(marker3);
	  delete(marker);
	  
	  pause(p.Results.time_interval);
      endswitch
      
      counter = counter + 1;
    endif
  endwhile
endfunction
