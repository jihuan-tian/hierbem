## Copyright (C) 2023-2024 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

function [M_cond, hmat_rel_err, x_octave, x_rel_err, hmat_factorized_rel_err] = process(trial_no)
  load(cstrcat("hmatrix-solve-lu-task-parallel-", num2str(trial_no), ".output"));
  load(cstrcat("M", num2str(trial_no), ".dat"));
  load(cstrcat("b", num2str(trial_no), ".dat"));

  ## Compute the condition number of the matrix. When it is large, the solution
  ## error will be higher.
  M_cond = cond(M)

  ## Calculate relative error between H-Matrix and full matrix based on Frobenius-norm
  hmat_rel_err = norm(H_full - M, 'fro') / norm(M, 'fro')

  ## Calculate relative error between H-matrix and full matrix solution based on 2-norm
  x_octave = M \ b;
  x_rel_err = norm(x_octave - x, 2) / norm(x_octave, 2)

  ## Compute the error between serial and parallel factorized H-matrices.
  hmat_factorized_rel_err = norm(LU_full - LU_full_serial, 'fro') / norm(LU_full_serial, 'fro')
endfunction
