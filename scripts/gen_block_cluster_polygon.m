function block_polygon = gen_block_cluster_polygon(tau_range, sigma_range, unit_size)
  yrange = [(tau_range(1) - 0.5) * unit_size, (tau_range(2) - 1 + 0.5) * unit_size] + 1;
  xrange = [(sigma_range(1) - 0.5) * unit_size, (sigma_range(2) - 1 + 0.5) * unit_size] + 1;

  block_polygon = [xrange(1), yrange(1);
		   xrange(1), yrange(2);
		   xrange(2), yrange(2);
		   xrange(2), yrange(1)];
endfunction
