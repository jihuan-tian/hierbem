function gen_matrix(trial_no)
  p = 5;
  n = 2^p;
  M = randn(n);
  M = M * M';
  b = randn(n, 1);
  
  save("-text", cstrcat("M", num2str(trial_no), ".dat"), "M");
  save("-text", cstrcat("b", num2str(trial_no), ".dat"), "b");
endfunction
