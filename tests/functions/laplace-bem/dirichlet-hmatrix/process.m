clear all;

load "matrices.dat";

figure;
show_matrix(K);
colormap hot;
plot_bct_struct("K_bct.dat");

figure;
show_matrix(V);
colormap hot;
plot_bct_struct("V_bct.dat");
