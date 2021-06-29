function plot_bct_struct(filename)
  pkg load matgeom;
  block_clusters = read_bct(filename);

  figure;
  hold on;
  axis equal;
  axis off;
  set(gca, "ydir", "reverse");
  
  for m = 1:length(block_clusters)
    plot_block_cluster(block_clusters{m}, 1);
  endfor
endfunction
