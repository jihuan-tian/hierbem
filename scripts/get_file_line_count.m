## Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function count = get_file_line_count(file_name)
  [status, output] = system(cstrcat('wc -l ', file_name));
  
  if (status == 0)
    count = str2num(strsplit(output, ' '){1});
  else
    count = -1;
  endif
endfunction
