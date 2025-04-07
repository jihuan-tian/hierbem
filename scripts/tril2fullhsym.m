function M_hsym = tril2fullhsym(L)
  ## tril2fullhsym - Convert a lower triangular matrix to a full
  ## Hermite symmetric matrix.

  [m, n] = size(L);
  if (m != n)
    error("Matrix row and column dimensions should be the same!");
  else
    ## Check if the imaginary parts of diagonal entries are zero.
    if sum(imag(diag(L))) != 0
      error("The diagonal entries of L should be real values.");
    else
      M_hsym = L + ctranspose(L - (eye(n) .* diag(L)));
    endif
  endif
endfunction
