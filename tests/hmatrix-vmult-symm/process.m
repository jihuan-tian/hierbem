clear all;
load M.dat;
load x.dat;
load hmatrix-vmult-symm.output;

MM=tril2fullsym(M);
HH_full=tril2fullsym(H_full);

printout_var("norm(HH_full-MM,'fro')/norm(MM,'fro')");
printout_var("norm(MM*x-y1,2)/norm(MM*x,2)");
printout_var("norm(HH_full*x-y1,2)/norm(HH_full*x,2)");
y1 ./ y2

figure;
hold on;
plot(y1, 'ro');
plot(MM*x, 'b+');
plot(HH_full*x,'kx');
