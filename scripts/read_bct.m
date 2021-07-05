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
      switch (length(fields))
	case 3
	  block_cluster = struct("tau", eval(fields{1}), "sigma", eval(fields{2}), "is_near_field", str2num(fields{3}));
	case 4
	  block_cluster = struct("tau", eval(fields{1}), "sigma", eval(fields{2}), "is_near_field", str2num(fields{3}), "rank", str2num(fields{4}));
      endswitch
      ## Append the block cluster to the result cell array.
      block_clusters{end+1} = block_cluster;
    endif
  endwhile
endfunction
