function save_distance_matrix(M, csv_filename)
  [m, n] = size(M);

  if (m != n)
    error("Matrix is not square!\n");
  endif

  number_of_dist_pairs = m * (m - 1) / 2;
  dist_pairs = zeros(number_of_dist_pairs, 1);

  counter = 1;
  for i = 2:m
    for j = 1:(i-1)
      dist_pairs(counter) = M(i, j);

      counter++;
    endfor
  endfor

  csvwrite(csv_filename, dist_pairs);
endfunction
