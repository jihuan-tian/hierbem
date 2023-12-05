load_packages;

figure();
plot(x, 'r-', "marker", "+");
hold on;
plot(x_octave, 'g-', "marker", "^");
hold off;

figure();
set_fig_size(gcf, [900, 400]);
subplot(1, 2, 1);
plot_bct_struct("H_bct.dat");
title("H-matrix");
subplot(1, 2, 2);
plot_bct_struct("LU_bct.dat");
title("LU factorization of H-matrix");
