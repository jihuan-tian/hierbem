function gen_matrix(trial_no)
  ## Generate the lower triangular part of the positive definite and
  ## symmetric matrix.
  p = 6;
  n = 2^p;
  M = randn(n);
  M = M * M';
  M = tril(M);
  b = randn(n, 1);
  
  save("-text", cstrcat("M", num2str(trial_no), ".dat"), "M");
  save("-text", cstrcat("b", num2str(trial_no), ".dat"), "b");
endfunction
