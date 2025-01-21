function x_permuted = permute_vector(x, permutation)
  x_permuted = zeros(size(x));

  for m = 1:length(x_permuted)
    x_permuted(m) = x(permutation(m));
  endfor
endfunction
