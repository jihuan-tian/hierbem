function y = backward_substitution(U, b)
  y = b;

  for m = size(U, 1):-1:1
    for n = (m+1):size(U,1)
      y(m) = y(m) - U(m,n) * y(n);
    endfor
    y(m) = y(m) / U(m,m);
  endfor
endfunction
