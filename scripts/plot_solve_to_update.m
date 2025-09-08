## Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your option)
## any later version. The full text of the license can be found in the file
## LICENSE at the top level directory of HierBEM.

function plot_solve_to_update(bct_file, task_edges_file, tau_range, sigma_range)
  ## Brackets should be escaped by backslash.
  output_file_basename = sprintf("solve-to-update-[%d,%d]-[%d,%d]", tau_range(1), tau_range(2), sigma_range(1), sigma_range(2));
  system(sprintf("grep -E \"^solve-(lower|upper)-to-update.*--> \\[%d,%d), \\[%d,%d)$\" %s > %s.txt", tau_range(1), tau_range(2), sigma_range(1), sigma_range(2), task_edges_file, output_file_basename));
  
  figure;
  plot_task_edges(bct_file, cstrcat(output_file_basename, ".txt"));
  PrintGCFLatex(cstrcat(output_file_basename, ".png"));
endfunction
