function plot_support_points(varargin)
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
  p.addParameter("marker", "o", @ischar);
  p.addParameter("markersize", 12, val_float);
  p.addParameter("markerfacecolor", "r", @ischar);
  p.addParameter("markeredgecolor", "r", @ischar);
  p.addParameter("labelsize", 18, val_float);

  p.parse(prop{:});

  x = [];
  y = [];
  labels = {};
  
  fid = fopen(p.Results.data_file, "r");
  while (true)
    line_str = fgetl(fid);

    if (line_str == -1)
      break;
    else
      line_str_split = strsplit(line_str, "\"");
      coords = textscan(line_str_split{1}, "%f %f");
      x(end+1) = coords{1};
      y(end+1) = coords{2};
      labels{end+1} = line_str_split{2};
    endif
  endwhile
  fclose(fid);

  n = length(x);

  ## Plot the data points.
  plot(x, y, "linestyle", "none", "marker", p.Results.marker, "markersize", p.Results.markersize, "markerfacecolor", p.Results.markerfacecolor, "markeredgecolor", p.Results.markeredgecolor);

  ## Add labels to the markers.
  hold on;
  for m = 1:n
    text(x(m) + p.Results.offx, y(m) + p.Results.offy, labels{m}, "fontsize", p.Results.labelsize);
  endfor

  ## When there is only one support point, also plot the four corner
  ## points of the unit square in order to make <code>axis
  ## equal</code> function properly.
  if (n == 1)
    plot([0, 1, 0, 1], [0, 0, 1, 1], "linestyle", "none", "marker", "none");
  endif

  axis equal;
  grid on;
  xticklabels([]);
  yticklabels([]);
  hold off;

  ## N.B. The adjustment of the @p xlim and @p ylim will not be
  ## effective if they are placed before axis equal.
  if (n == 1)
    xlim([0, 1]);
    ylim([0, 1]);
  else
    xlim([min(x), max(x)]);
    ylim([min(y), max(y)]);
  endif
endfunction
