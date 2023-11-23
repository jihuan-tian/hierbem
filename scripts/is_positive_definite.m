function ret = is_positive_definite(M)
  lambda  = eig(M);
  min_lambda = min(lambda);
  if (min_lambda < 0)
    fprintf(stdout(),
	    "The matrix is not positive definitive with the minimum eigen value %g!\n", min_lambda);

    ret = false;
  else
    fprintf(stdout(),
	    "The matrix is positive definitive with the minimum eigen value %g!\n", min_lambda);

    ret = true;
  endif
endfunction
