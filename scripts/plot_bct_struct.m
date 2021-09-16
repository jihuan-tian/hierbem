function plot_bct_struct(filename, enable_display_rank)
  if (!exist("enable_display_rank", "var"))
    enable_display_rank = true;
  endif

  pkg load matgeom;
  block_clusters = read_bct(filename);

  hold on;
  axis equal;
  axis off;
  set(gca, "ydir", "reverse");
  
  for m = 1:length(block_clusters)
    plot_block_cluster(block_clusters{m}, 1, enable_display_rank);
  endfor

  hold off;
endfunction
