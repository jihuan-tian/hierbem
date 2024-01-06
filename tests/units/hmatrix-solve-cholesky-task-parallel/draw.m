function draw(trial_no)
  load_packages;
  load(cstrcat("hmatrix-solve-cholesky-task-parallel-", num2str(trial_no), ".output"));
  [hmat_rel_err, product_hmat_rel_err, x_octave, x_rel_err, hmat_factorized_rel_err] = process(trial_no);

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
