function plot_bct_struct(filename, enable_display_rank, is_fill)
  if (!exist("enable_display_rank", "var"))
    enable_display_rank = true;
  endif

  if (!exist("is_fill", "var"))
    is_fill = true;
  endif

  pkg load matgeom;
  block_clusters = read_bct(filename);

  hold on;
  
  for m = 1:length(block_clusters)
    plot_block_cluster(block_clusters{m}, 1, enable_display_rank, is_fill);
  endfor

  axis equal;
  axis off;
  set(gca, "ydir", "reverse");

  hold off;
endfunction
