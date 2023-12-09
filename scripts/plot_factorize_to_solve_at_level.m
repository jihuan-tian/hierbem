function plot_factorize_to_solve_at_level(bct_file, task_edges_file, level)
  output_file_basename = sprintf("factorize-to-solve-level%d", level);
  system(sprintf("grep -E \"^factorize-to-solve-(lower|upper) at level %d\" %s > %s.txt", level, task_edges_file, output_file_basename));

  figure;
  plot_task_edges(bct_file, cstrcat(output_file_basename, ".txt"));
  PrintGCFLatex(cstrcat(output_file_basename, ".png"));
endfunction
