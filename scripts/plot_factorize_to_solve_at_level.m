## Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function plot_factorize_to_solve_at_level(bct_file, task_edges_file, level)
  output_file_basename = sprintf("factorize-to-solve-level%d", level);
  system(sprintf("grep -E \"^factorize-to-solve-(lower|upper) at level %d\" %s > %s.txt", level, task_edges_file, output_file_basename));

  figure;
  plot_task_edges(bct_file, cstrcat(output_file_basename, ".txt"));
  PrintGCFLatex(cstrcat(output_file_basename, ".png"));
endfunction
