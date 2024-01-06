figure;
plot(x,'ro');
hold on;
plot(x_octave,'b+');
hold off;

figure;
set_fig_size(gcf, 900, 400);
subplot(1,2,1);
plot_bct_struct("H_bct.dat");
title("H-matrix");
subplot(1,2,2);
plot_bct_struct("L_bct.dat");
title("Cholesky factorization of H-matrix");

figure;
show_matrix(L_full);
hold on;
plot_bct_struct("L_bct.dat", false, false);
