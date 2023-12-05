load_packages;

## Plot the error of the solution matrix.
figure;
show_matrix(X - X_octave);

## Plot the block cluster tree structures.
figure;
set_fig_size(gcf, [900, 300]);
subplot(1, 3, 1);
plot_bct_struct("HL_bct.dat");
title("Lower triangular matrix");
subplot(1, 3, 2);
plot_bct_struct("HZ_bct.dat");
title("RHS matrix");
subplot(1, 3, 3);
plot_bct_struct("HX_bct.dat");
title("Solution matrix");
