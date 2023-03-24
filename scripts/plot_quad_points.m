function plot_quad_points(varargin)
  [reg, prop] = parseparams(varargin);

  switch (length(reg) > 0)
    case 1
      h = gcf;
      points = reg{1};
    case 2
      h = reg{1};
      points = reg{2};
    otherwise
      error("Wrong number of arguments!");
  endswitch

  p = inputParser();
  p.FunctionName = "plot_quad_points";
  val_float = @(x) isscalar(x) && isfloat(x);
  p.addParameter("offx", 0.02, val_float);
  p.addParameter("offy", 0.02, val_float);
  p.addParameter("offz", 0.02, val_float);
  p.addParameter("marker", "o", @ischar);
  p.addParameter("markersize", 12, val_float);
  p.addParameter("markerfacecolor", "r", @ischar);
  p.addParameter("markeredgecolor", "r", @ischar);
  p.addParameter("labelsize", 18, val_float);

  p.parse(prop{:});

  [point_num, dim] = size(points);

  axis_length = 0.2;
  switch(dim)
    case 2
      plot(points(:, 1), points(:, 2), "linestyle", "none", "marker", p.Results.marker, "markersize", p.Results.markersize, "markerfacecolor", p.Results.markerfacecolor, "markeredgecolor", p.Results.markeredgecolor);
      plot_axes([min(points(:, 1)) + max(points(:, 1)), min(points(:, 2)) + max(points(:, 2))] / 2, axis_length);
    case 3
      plot3(points(:, 1), points(:, 2), points(:, 3), "linestyle", "none", "marker", p.Results.marker, "markersize", p.Results.markersize, "markerfacecolor", p.Results.markerfacecolor, "markeredgecolor", p.Results.markeredgecolor);
      plot_axes([min(points(:, 1)) + max(points(:, 1)), min(points(:, 2)) + max(points(:, 2)), min(points(:, 3)) + max(points(:, 3))] / 2, axis_length);
    otherwise
      error("Point dimension not supported!");
  endswitch

  ## Add labels to the markers.
  hold on;
  if (p.Results.labelsize > 0)
    for m = 1:point_num
      switch (dim)
	case 2
	  text(points(m, 1) + p.Results.offx, points(m, 2) + p.Results.offy, num2str(m), "fontsize", p.Results.labelsize);
	case 3
	  text(points(m, 1) + p.Results.offx, points(m, 2) + p.Results.offy, points(m, 3) + p.Results.offz, num2str(m), "fontsize", p.Results.labelsize);
	otherwise
	  error("Point dimension not supported!");
      endswitch
    endfor
  endif

  if (point_num == 1)
    switch(dim)
      case 2
	## When there is only one support point, also plot the four corner
	## points of the unit square in order to make <code>axis
	## equal</code> function properly.
	plot([0, 1, 0, 1], [0, 0, 1, 1], "linestyle", "none", "marker", "none");
      case 3
	## When there is only one support point, also plot the eight corner
	## points of the unit square in order to make <code>axis
	## equal</code> function properly.
	plot3([0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], "linestyle", "none", "marker", "none");
      otherwise
	error("Point dimension not supported!");
    endswitch
  endif

  axis equal;
  ## Show the full surrounding box of the axes.
  if (dim == 3)
    set(gca, "box", "on");
    set(gca, "boxstyle", "full");
  endif

  grid on;
  xticklabels([]);
  yticklabels([]);
  if (dim == 3)
    zticklabels([]);
  endif
  hold off;

  ## N.B. The adjustment of the @p xlim and @p ylim will not be
  ## effective if they are placed before axis equal.
  if (point_num == 1)
    xlim([0, 1]);
    ylim([0, 1]);
    if (dim == 3)
      zlim([0, 1]);
    endif
  else
    xlim([min(points(:, 1)), max(points(:, 1))]);
    ylim([min(points(:, 2)), max(points(:, 2))]);
    if (dim == 3)
      if (min(z) != max(z))
	zlim([min(points(:, 3)), max(points(:, 3))]);
      endif
    endif
  endif

  xlabel("X");
  ylabel("Y");
  set(gca, "xtick", []);
  set(gca, "ytick", []);
  
  if (dim == 3)
    zlabel("Z");
    set(gca, "ztick", []);
  endif

  ## Enable the default figure toolbar.
  set(h, "toolbar", "figure");

  ## Create a edit box for input of point indices. Create a toggle button for
  ## showing and hiding the text labels.
  uictrl_height = 30;
  uictrl_width = 60;
  uictrl_vspace = 5;
  uictrl_num = 4;

  panel = uipanel(h, "units", "pixels", "title", "Mark point", "titleposition", "centertop", "position", [5, 5, uictrl_width * 1.1, uictrl_height * uictrl_num + uictrl_vspace * (uictrl_num - 1)]);
  
  point_box = uicontrol(panel, "style", "edit", "backgroundcolor", [0.75,0.75,0.75], "position", [0, (uictrl_height + uictrl_vspace) * (uictrl_num - 2), uictrl_width, uictrl_height], "tooltipstring", "Input point indices separated by comma");
  switch(dim)
    case 2
      draw_btn = uicontrol(panel, "style", "pushbutton", "position", [0, (uictrl_height + uictrl_vspace) * (uictrl_num - 3), uictrl_width, uictrl_height], "string", "Draw", "callback", {@mark_point_in_point_box, get(h, "currentaxes"), point_box, points(:, 1), points(:, 2), []});
    case 3
      draw_btn = uicontrol(panel, "style", "pushbutton", "position", [0, (uictrl_height + uictrl_vspace) * (uictrl_num - 3), uictrl_width, uictrl_height], "string", "Draw", "callback", {@mark_point_in_point_box, get(h, "currentaxes"), point_box, points(:, 1), points(:, 2), points(:, 3)});
    otherwise
      error("Point dimension not supported!");
  endswitch
  clear_btn = uicontrol(panel, "style", "pushbutton", "position", [0, (uictrl_height + uictrl_vspace) * (uictrl_num - 4), uictrl_width, uictrl_height], "string", "Clear", "callback", {@clear_point, get(h, "currentaxes")});
endfunction
