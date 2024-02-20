clear all;

load_packages;

figure;
subplot(1, 3, 1);
plot_bct_struct("bct1.dat", 'show_rank', false);
subplot(1, 3, 2);
plot_bct_struct("bct2.dat", 'show_rank', false);
subplot(1, 3, 3);
plot_bct_struct("bct3.dat", 'show_rank', false);
