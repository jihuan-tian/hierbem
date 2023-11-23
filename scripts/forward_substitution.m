function y = forward_substitution(L, b)
  y = b;

  for m = 1:size(L, 1)
    for n = 1:m-1
      y(m) = y(m) - L(m,n) * y(n);
    endfor
    y(m) = y(m) / L(m,m);
  endfor
endfunction
