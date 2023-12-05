clear all;

load_packages;

figure;
subplot(3, 2, 1);
plot_bct_struct("bct1.dat");
subplot(3, 2, 2);
plot_bct_struct("bct2.dat");
subplot(3, 2, 3);
plot_bct_struct("bct3.dat");
subplot(3, 2, 4);
plot_bct_struct("bct4.dat");
subplot(3, 2, 5);
plot_bct_struct("bct5.dat");

PrintGCF("bct-hp-fine");
