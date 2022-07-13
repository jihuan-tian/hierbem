function [x, y, z, labels] = plot_support_points(varargin)
  ## plot_support_points - Plot the support points in a cell generated
  ## by @p DoFTools::map_dofs_to_support_points in deal.ii for a
  ## single cell mesh.

  ## The function @p parseparams is used for splitting the argument
  ## list into figure handle part and the remaining part.
  [reg, prop] = parseparams(varargin);

  if (length(reg) > 0)
    h = reg(1);
  else
    h = gcf;
  endif

  p = inputParser();
  p.FunctionName = "plot_support_points";
  p.addRequired("data_file", @ischar);
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

  x = [];
  y = [];
  z = [];
  labels = {};
  coordinates_num = 0;
  
  fid = fopen(p.Results.data_file, "r");
  while (true)
    line_str = fgetl(fid);

    if (line_str == -1)
      break;
    else
      line_str_split = strsplit(line_str, "\"");

      ## Check how many fields are there in the split line string.
      coordinates_split_str = strsplit(strtrim(line_str_split{1}), " ");
      coordinates_num = length(coordinates_split_str);
      
      switch (coordinates_num)
	case 2
	  coords = textscan(line_str_split{1}, "%f %f");

	  x(end+1) = coords{1};
	  y(end+1) = coords{2};
	case 3
	  coords = textscan(line_str_split{1}, "%f %f %f");

	  x(end+1) = coords{1};
	  y(end+1) = coords{2};
	  z(end+1) = coords{3};
	otherwise
	  error(cstrcat("There are ", num2str(coordinates_num), " coordinates, which is not supported!"));
      endswitch
      
      if (length(line_str_split) >= 2)
	labels{end+1} = line_str_split{2};
      endif
    endif
  endwhile
  fclose(fid);

  n = length(x);

  ## Plot the data points and axes.
  axis_length = 0.2;
  switch(coordinates_num)
    case 2
      plot(x, y, "linestyle", "none", "marker", p.Results.marker, "markersize", p.Results.markersize, "markerfacecolor", p.Results.markerfacecolor, "markeredgecolor", p.Results.markeredgecolor);
      plot_axes([min(x) + max(x), min(y) + max(y)] / 2, axis_length);
    case 3
      plot3(x, y, z, "linestyle", "none", "marker", p.Results.marker, "markersize", p.Results.markersize, "markerfacecolor", p.Results.markerfacecolor, "markeredgecolor", p.Results.markeredgecolor);
      plot_axes([min(x) + max(x), min(y) + max(y), min(z) + max(z)] / 2, axis_length);
  endswitch

  ## Add labels to the markers.
  hold on;
  if (p.Results.labelsize > 0 && length(labels) > 0)
    for m = 1:n
      switch (coordinates_num)
	case 2
	  text(x(m) + p.Results.offx, y(m) + p.Results.offy, labels{m}, "fontsize", p.Results.labelsize);
	case 3
	  text(x(m) + p.Results.offx, y(m) + p.Results.offy, z(m) + p.Results.offz, labels{m}, "fontsize", p.Results.labelsize);
      endswitch
    endfor
  endif

  if (n == 1)
    switch(coordinates_num)
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
    endswitch
  endif

  axis equal;
  ## Show the full surrounding box of the axes.
  if (coordinates_num == 3)
    set(gca, "box", "on");
    set(gca, "boxstyle", "full");
  endif

  grid on;
  xticklabels([]);
  yticklabels([]);
  if (coordinates_num == 3)
    zticklabels([]);
  endif
  hold off;

  ## N.B. The adjustment of the @p xlim and @p ylim will not be
  ## effective if they are placed before axis equal.
  if (n == 1)
    xlim([0, 1]);
    ylim([0, 1]);
    if (coordinates_num == 3)
      zlim([0, 1]);
    endif
  else
    xlim([min(x), max(x)]);
    ylim([min(y), max(y)]);
    if (coordinates_num == 3)
      if (min(z) != max(z))
	zlim([min(z), max(z)]);
      endif
    endif
  endif

  xlabel("X");
  ylabel("Y");
  set(gca, "xtick", []);
  set(gca, "ytick", []);
  
  if (coordinates_num == 3)
    zlabel("Z");
    set(gca, "ztick", []);
  endif

  ## Enable the default figure toolbar.
  set(h, "toolbar", "figure");

  ## Create a edit box for input of DoF indices. Create a toggle button for
  ## showing and hiding the text labels.
  uictrl_height = 30;
  uictrl_width = 60;
  uictrl_vspace = 5;
  uictrl_num = 4;

  panel = uipanel(h, "units", "pixels", "title", "Mark DoF", "titleposition", "centertop", "position", [5, 5, uictrl_width * 1.1, uictrl_height * uictrl_num + uictrl_vspace * (uictrl_num - 1)]);
  
  dof_box = uicontrol(panel, "style", "edit", "backgroundcolor", [0.75,0.75,0.75], "position", [0, (uictrl_height + uictrl_vspace) * (uictrl_num - 2), uictrl_width, uictrl_height], "tooltipstring", "Input DoF indices separated by comma");
  draw_btn = uicontrol(panel, "style", "pushbutton", "position", [0, (uictrl_height + uictrl_vspace) * (uictrl_num - 3), uictrl_width, uictrl_height], "string", "Draw", "callback", {@mark_dof, get(h, "currentaxes"), dof_box, x, y, z, labels});
  clear_btn = uicontrol(panel, "style", "pushbutton", "position", [0, (uictrl_height + uictrl_vspace) * (uictrl_num - 4), uictrl_width, uictrl_height], "string", "Clear", "callback", {@clear_dof, get(h, "currentaxes")});
endfunction

function mark_dof(h, evt, current_axes, dof_box, x, y, z, labels)
  hold(current_axes, "on");

  dof_index_str = get(dof_box, "string");
  dof_index = eval(cstrcat("[", dof_index_str, "]"));

  for l = 1:length(dof_index)
    for m = 1:length(x)
      current_label = cellfun(@str2num, strsplit(labels{m}, ","), "UniformOutput", true);
      idx = find(current_label == dof_index(l));

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

function clear_dof(h, evt, current_axes)
  list_of_markder_handles = get(current_axes, "userdata");

  if (length(list_of_markder_handles) > 0)
    delete(list_of_markder_handles);
    set(current_axes, "userdata", []);
  endif
endfunction
