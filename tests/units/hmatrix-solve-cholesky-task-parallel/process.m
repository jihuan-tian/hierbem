function [MM_cond, hmat_rel_err, product_hmat_rel_err, x_octave, x_rel_err, hmat_factorized_rel_err] = process(trial_no)
  load(cstrcat("hmatrix-solve-cholesky-task-parallel-", num2str(trial_no), ".output"));
  load(cstrcat("M", num2str(trial_no), ".dat"));
  load(cstrcat("b", num2str(trial_no), ".dat"));

  ## Restore the matrix M to be symmetric.
  MM = tril2fullsym(M);
  ## Compute its condition number. When it is large, the solution error will be
  ## higher.
  MM_cond = cond(MM)

  ## Calculate relative error between H-Matrix and full matrix based on
  ## Frobenius-norm. Only the lower triangular part is compared.
  hmat_rel_err = norm(H_full - M, 'fro') / norm(M, 'fro')

  ## Calculate the relative error between L*L^T and the original symmetric matrix.
  product_hmat_rel_err = norm(L_full*L_full' - MM, 'fro') / norm(MM, 'fro')

  ## Calculate the relative error between H-matrix and full matrix solution
  ## vectors.
  x_octave = MM \ b;
  x_rel_err = norm(x_octave - x, 2) / norm(x_octave, 2)

  ## Compute the error between serial and parallel factorized H-matrices.
  hmat_factorized_rel_err = norm(L_full - L_full_serial, 'fro') / norm(L_full_serial, 'fro')
endfunction
