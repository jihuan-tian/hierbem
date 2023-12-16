function count = get_file_line_count(file_name)
  [status, output] = system(cstrcat('wc -l ', file_name));
  
  if (status == 0)
    count = str2num(strsplit(output, ' '){1});
  else
    count = -1;
  endif
endfunction
