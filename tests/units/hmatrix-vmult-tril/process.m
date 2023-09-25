clear all;
load M.dat;
load x.dat;
load hmatrix-vmult-tril.output;

printout_var("norm(H_full-M,'fro')/norm(M,'fro')");
printout_var("norm(M*x-y1,2)/norm(M*x,2)");
printout_var("norm(H_full*x-y1,2)/norm(H_full*x,2)");
y1 ./ y2

figure;
hold on;
plot(y1, 'ro');
plot(M*x, 'b+');
plot(H_full*x,'kx');
