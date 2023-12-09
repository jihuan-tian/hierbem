function plot_solve_to_update(bct_file, task_edges_file, tau_range, sigma_range)
  ## Brackets should be escaped by backslash.
  output_file_basename = sprintf("solve-to-update-[%d,%d]-[%d,%d]", tau_range(1), tau_range(2), sigma_range(1), sigma_range(2));
  system(sprintf("grep -E \"^solve-(lower|upper)-to-update.*--> \\[%d,%d), \\[%d,%d)$\" %s > %s.txt", tau_range(1), tau_range(2), sigma_range(1), sigma_range(2), task_edges_file, output_file_basename));
  
  figure;
  plot_task_edges(bct_file, cstrcat(output_file_basename, ".txt"));
  PrintGCFLatex(cstrcat(output_file_basename, ".png"));
endfunction
