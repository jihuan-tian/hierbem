function M_sym = tril2fullsym(L)
  ## tril2fullsym - Convert a lower triangular matrix to a full
  ## symmetric matrix.

  [m, n] = size(L);
  if (m != n)
    error("Matrix row and column dimensions should be the same!");
  else
    M_sym = L + (L - (eye(n) .* diag(L)))';
  endif
endfunction
