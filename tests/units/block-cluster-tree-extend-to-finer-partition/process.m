clear all;

load_packages;

figure;
subplot(2,2,1);
plot_bct_struct("bct1.dat", false);
subplot(2,2,2);
plot_bct_struct("bct2.dat", false);
subplot(2,2,3);
plot_bct_struct("bct1_ext.dat", false);
subplot(2,2,4);
plot_bct_struct("bct2_ext.dat", false);

PrintGCF("bct");
