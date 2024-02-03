function draw(trial_no)
  load_packages;
  load(cstrcat("hmatrix-solve-lu-task-parallel-", num2str(trial_no), ".output"));
  [M_cond, hmat_rel_err, x_octave, x_rel_err, hmat_factorized_rel_err] = process(trial_no);

  figure;
  hold on;
  plot(x_octave, 'b.');
  plot(x, 'ro');
  legend("x octave", "x cpp");
  title("Solution vector");

  figure;
  plot_bct_struct("H_bct.dat");
  title("Block structure");

  figure;
  show_matrix(LU_full - LU_full_serial);
  hold on;
  plot_bct_struct("H_bct.dat", "show_rank", false, "fill_block", false, "border_color", "k");
  title("Absolute error of factorized H-matrix");

  figure;
  show_matrix(LU_full);
  title("LU factors");
endfunction
