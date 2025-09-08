## Copyright (C) 2024 Jihuan Tian <jihuan_tian@hotmail.com>
##
## This file is part of the HierBEM library.
##
## HierBEM is free software: you can use it, redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License as published by the
## Free Software Foundation, either version 3 of the License, or (at your
## option) any later version. The full text of the license can be found in the
## file LICENSE at the top level directory of HierBEM.

function draw(trial_no)
  load_packages;
  load(cstrcat("hmatrix-solve-cholesky-task-parallel-", num2str(trial_no), ".output"));
  [MM_cond, hmat_rel_err, product_hmat_rel_err, x_octave, x_rel_err, hmat_factorized_rel_err] = process(trial_no);

  figure;
  plot(x,'ro');
  hold on;
  plot(x_octave,'b+');
  hold off;

  figure;
  show_matrix(L_full);
  hold on;
  plot_bct_struct("H_bct.dat", 'fill_block', false, 'near_field_rank_color', 'white', 'far_field_rank_color', 'white');
endfunction
