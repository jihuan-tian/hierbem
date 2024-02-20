clear all;

load_packages;

figure;
subplot(3, 2, 1);
plot_bct_struct("bct1.dat", 'show_rank', false);
subplot(3, 2, 2);
plot_bct_struct("bct2.dat", 'show_rank', false);
subplot(3, 2, 3);
plot_bct_struct("bct3.dat", 'show_rank', false);
subplot(3, 2, 4);
plot_bct_struct("bct4.dat", 'show_rank', false);
subplot(3, 2, 5);
plot_bct_struct("bct5.dat", 'show_rank', false);
