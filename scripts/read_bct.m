function block_clusters =  read_bct(filename)
  ## read_bct - Read the leaf set of a block cluster tree.

  fid = fopen(filename, "r");
  
  block_clusters = cell();
  while (true)
    line_str = fgetl(fid);
    
    if (line_str == -1)
      break;
    else
      fields = strsplit(line_str, ",");
      tau_range = strrep(fields{1}, ")", "]");
      sigma_range = strrep(fields{2}, ")", "]");
      switch (length(fields))
	case 3
	  ## When matrix ranks are not computed.
	  block_cluster = struct("tau", eval(tau_range), "sigma", eval(sigma_range), "is_near_field", str2num(fields{3}));
	case 4
	  ## When matrix ranks are computed.
	  block_cluster = struct("tau", eval(tau_range), "sigma", eval(sigma_range), "is_near_field", str2num(fields{3}), "rank", str2num(fields{4}));
      endswitch
      ## Append the block cluster to the result cell array.
      block_clusters{end+1} = block_cluster;
    endif
  endwhile
endfunction
