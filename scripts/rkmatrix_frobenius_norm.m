function result = rkmatrix_frobenius_norm(A, B)
  k = size(A, 2);
  if (k != size(B, 2))
    error("Matrix A and B should have the same number of columns");
  endif

  result = 0;
  
  for m = 1:k
    for n = 1:k
      result = result + sum(A(:,m) .* conj(A(:,n))) * sum(B(:,m) .* conj(B(:,n)));
    endfor
  endfor
endfunction
