clear all;

load_packages;

figure;
subplot(1, 2, 1);
plot_bct_struct("h1.dat");
PrintGCF("h1.png");
subplot(1, 2, 2);
plot_bct_struct("h2.dat");
PrintGCF("h2.png");
