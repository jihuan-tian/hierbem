function sp = gen_permutation_matrix(permutation)
  original_indices = reshape(1:length(permutation), size(permutation, 1), size(permutation, 2));
  values = ones(size(permutation));
  sp = sparse(permutation, original_indices, values);
endfunction
